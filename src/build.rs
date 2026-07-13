//! Streaming trigram-index build path (external sort + k-way merge).
//!
//! `fgr index` previously held the entire trigram store in a `HashMap` and
//! serialized it in one shot (`SparseIndex::build_from_directory`). That made
//! peak RAM scale with total posting-byte volume, so AOSP-sized corpora OOM'd.
//!
//! This module replaces the in-RAM accumulation with a two-phase pipeline:
//!
//! 1. **Extraction + spill**: chunk the corpus (≤ `chunk_files` files or ≤
//!    `chunk_byte_target` bytes of raw file content, whichever first). For
//!    each chunk, read files in parallel, extract trigrams into a
//!    `ChunkOutputs` (case-sensitive + optional case-insensitive companion),
//!    sort the postings by `(tri_hash, doc_id, line_no, byte_offset)`, and
//!    serialize one postings spill + one masks spill per chunk to a temp
//!    directory. Each chunk's working set is released before the next starts.
//! 2. **K-way merge (M2)**: open all spill fronts in a min-heap, emit the
//!    final `ngrams.postings` / `ngrams.bitmaps` / `ngrams.masks` files in
//!    `crc32fast::hash(tri)` ascending order across all three. Peak RAM
//!    becomes: one chunk's working set + one trigram's posting buffer (capped
//!    by `max_posting_buf_bytes`) + one trigram's Roaring bitmap + one
//!    `FollowerMask`.
//!
//! This file holds M1: the extractor, `ChunkOutputs`, and the spill
//! readers/writers. M2 (k-way merge) and M3 (glue: `streaming_build` entry
//! points) are added incrementally below.
//!
//! Invariants preserved from the legacy `SparseIndex::add_document` path
//! (see `src/index.rs`):
//!
//! - Dedup key is `(trigram, doc_id, line_no)`. The legacy path deduped
//!   per-trigram via `TrigramBuilder.writer.last_dl()`; the streaming path
//!   pushes every window occurrence and dedupes after sort (adjacent equal
//!   keys in the spill).
//! - The CS mask pass uses a 4-byte window over the whole file, not per
//!   line, so a trigram straddling a line break still records its line-broader
//!   follower (regression-tested in `src/index.rs:411-420`).
//! - CS+CI lockstep: every byte is visited once for the CS pass and once for
//!   the CI pass in the same `extract_file` call, sharing the per-line
//!   `fold_buf` scratch.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::casefold;
use crate::index::FollowerMask;

// ─── Public types ─────────────────────────────────────────────────────────────

/// One trigram occurrence extracted from a chunk. The streaming build sorts a
/// `Vec<PostingRecord>` by `(tri_hash, doc_id, line_no, byte_offset)` and
/// dedupes adjacent equal keys — every window occurrence is emitted first so
/// the in-flight dedup work happens off the hot extraction path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostingRecord {
    /// `crc32fast::hash(tri)` — used as the sort key and the eventual on-disk
    /// lookup-key.
    pub tri_hash: u32,
    pub doc_id: u32,
    pub line_no: u32,
    /// Start of the line in the file (all trigrams within the same line share
    /// `byte_offset`).
    pub byte_offset: u32,
}

impl PostingRecord {
    /// 16 bytes (4 × u32).
    #[allow(dead_code)] // public wire-format constant for callers / tests.
    pub const SIZE: usize = 16;
}

/// Accumulator that `extract_file` fills per chunk. CS fields are always
/// populated; CI fields stay empty when the build is case-sensitive only.
#[derive(Default)]
pub struct ChunkOutputs {
    pub postings: Vec<PostingRecord>,
    pub masks: HashMap<[u8; 3], FollowerMask>,
    pub postings_ci: Vec<PostingRecord>,
    pub masks_ci: HashMap<[u8; 3], FollowerMask>,
}

impl ChunkOutputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.postings.clear();
        self.masks.clear();
        self.postings_ci.clear();
        self.masks_ci.clear();
    }
}

// ─── Extraction ───────────────────────────────────────────────────────────────

/// Extract trigram postings (CS + optional CI) and successor masks from one
/// document. Caller-provided `chunk_out` is mutated in place.
///
/// `doc_id` is the absolute document index within the corpus (assigned by the
/// chunk's ingest loop). `byte_offset` recorded per posting is the byte offset
/// of the start of the line within the document.
pub fn extract_file(
    content: &[u8],
    doc_id: u32,
    case_insensitive: bool,
    chunk_out: &mut ChunkOutputs,
) {
    if content.len() < 3 {
        return;
    }

    // CS whole-file mask pass: 4-byte window over the entire file so a trigram
    // straddling a line break still records its line-broader follower.
    for w in content.windows(4) {
        let tri = [w[0], w[1], w[2]];
        let follower = w[3];
        let mask = chunk_out.masks.entry(tri).or_insert([0u32; 8]);
        set_mask_bit(mask, follower);
    }

    // Per-line trigram posting pass (CS).
    let mut line_no = 1u32;
    let mut line_start = 0usize;
    let mut fold_buf: Vec<u8> = Vec::new();

    loop {
        let line_end = content[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| line_start + p)
            .unwrap_or(content.len());
        let line = &content[line_start..line_end];

        if line.len() >= 3 {
            let byte_offset = line_start as u32;
            chunk_out.postings.reserve(line.len() - 2);
            for w in line.windows(3) {
                let tri = [w[0], w[1], w[2]];
                chunk_out.postings.push(PostingRecord {
                    tri_hash: crc32fast::hash(&tri),
                    doc_id,
                    line_no,
                    byte_offset,
                });
            }

            if case_insensitive {
                casefold::fold_into(line, &mut fold_buf);
                if fold_buf.len() >= 3 {
                    chunk_out.postings_ci.reserve(fold_buf.len() - 2);
                    for w in fold_buf.windows(3) {
                        let tri = [w[0], w[1], w[2]];
                        chunk_out.postings_ci.push(PostingRecord {
                            tri_hash: crc32fast::hash(&tri),
                            doc_id,
                            line_no,
                            byte_offset,
                        });
                    }
                }
            }
        }

        if line_end >= content.len() {
            break;
        }
        line_start = line_end + 1;
        line_no += 1;
    }

    // CI whole-file mask pass: fold once over the whole file, then run the
    // 4-byte window pass against the folded buffer.
    if case_insensitive {
        let mut file_fold: Vec<u8> = Vec::new();
        casefold::fold_into(content, &mut file_fold);
        if file_fold.len() >= 4 {
            for w in file_fold.windows(4) {
                let tri = [w[0], w[1], w[2]];
                let follower = w[3];
                let mask = chunk_out.masks_ci.entry(tri).or_insert([0u32; 8]);
                set_mask_bit(mask, follower);
            }
        }
    }
}

#[inline]
fn set_mask_bit(mask: &mut FollowerMask, follower: u8) {
    let word = (follower >> 5) & 0x7;
    let bit = follower & 0x1F;
    mask[word as usize] |= 1u32 << bit;
}

// ─── Spill file format ────────────────────────────────────────────────────────
//
// Postings spill ("FSPP"):
//   u32 magic           = 0x46535050
//   u32 version         = 1
//   u64 record_count
//   record_count × (u32 tri_hash, u32 doc_id, u32 line_no, u32 byte_offset)
//   = 16 + 16·N bytes. Sorted ascending by (tri_hash, doc_id, line_no, byte_offset).
//
// Masks spill ("FSMP"):
//   u32 magic           = 0x46534D50
//   u32 version         = 1
//   N × (u32 tri_hash, [u32;8] mask_words)
//   = 8 + 36·N bytes. Sorted ascending by tri_hash. A trigram with no
//   recorded follower still gets an entry — its mask is all zeros, so the
//   on-disk binary-search across postings/lookups stays aligned (the same
//   invariant the legacy `write_ngram_files` preserves at
//   `src/persist.rs:1030-1042`).

const POSTINGS_SPILL_MAGIC: u32 = 0x4653_5050; // "FSPP" little-endian
const MASKS_SPILL_MAGIC: u32 = 0x4653_4D50; // "FSMP" little-endian
const SPILL_VERSION: u32 = 1;

// ─── SpillDir RAII ────────────────────────────────────────────────────────────

/// RAII wrapper around a temp directory that holds chunk spill files. Drops
/// the directory and all files inside on scope exit (success or panic).
///
/// Convention: `std::env::temp_dir().join("fgr-build-{pid}-{counter}")` with an
/// auto-incrementing counter to avoid name collisions across concurrent
/// builds. Files inside this dir are named by the spill writer (`PostingsSpill
/// ::new_path(dir, seq)` and `MasksSpill::new_path(dir, seq)`).
pub struct SpillDir {
    path: PathBuf,
    // Keep the handles alive so the OS does not delete files mid-write; on
    // Drop we also explicitly `remove_dir_all` as a belt-and-suspenders
    // measure for platforms where `tempdir`'s background reaper is racy.
    _cleanup: SpillCleanup,
}

struct SpillCleanup {
    path: PathBuf,
}

impl Drop for SpillCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

static SPILL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl SpillDir {
    pub fn new() -> Result<Self> {
        let n = SPILL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "fgr-build-{}-{}-{}",
            std::process::id(),
            n,
            uuid_like_suffix(),
        ));
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating spill dir {}", path.display()))?;
        let cleanup_path = path.clone();
        Ok(Self {
            path,
            _cleanup: SpillCleanup { path: cleanup_path },
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// 8-char hex suffix derived from a thread-local counter (no UUID dep). Unique
/// enough for the spill-dir naming collision space (we only need uniqueness
/// among concurrent `fgr` processes on this host).
fn uuid_like_suffix() -> String {
    use std::sync::atomic::Ordering;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let s = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{:08x}", s & 0xFFFF_FFFF)
}

// ─── PostingsSpill ────────────────────────────────────────────────────────────

/// Writer over a `.postings.spill` file. Caller pushes records; the writer
/// appends the header on `finish()` and fsyncs.
pub struct PostingsSpillWriter {
    file: BufWriter<File>,
    count: u64,
}

impl PostingsSpillWriter {
    /// `buf_size` is the `BufWriter` capacity for this spill. Postings are
    /// written in 16-byte records so the std default (8 KiB) fires a syscall
    /// every ~512 records; passing at least 256 KiB is a major throughput
    /// win on large corpora. See `StreamingConfig::write_buffer_bytes`.
    pub fn create(path: &Path, buf_size: usize) -> Result<Self> {
        let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = BufWriter::with_capacity(buf_size, f);
        w.write_u32::<LittleEndian>(POSTINGS_SPILL_MAGIC)?;
        w.write_u32::<LittleEndian>(SPILL_VERSION)?;
        // Reserve space for record_count; fill in on `finish`.
        w.write_u64::<LittleEndian>(0)?;
        Ok(Self { file: w, count: 0 })
    }

    pub fn push(&mut self, rec: PostingRecord) -> Result<()> {
        self.file.write_u32::<LittleEndian>(rec.tri_hash)?;
        self.file.write_u32::<LittleEndian>(rec.doc_id)?;
        self.file.write_u32::<LittleEndian>(rec.line_no)?;
        self.file.write_u32::<LittleEndian>(rec.byte_offset)?;
        self.count += 1;
        Ok(())
    }

    #[allow(dead_code)] // public batch helper for callers emitting many records.
    pub fn push_slice(&mut self, recs: &[PostingRecord]) -> Result<()> {
        for r in recs {
            self.push(*r)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        let n = self.count;
        self.file.flush()?;
        let mut f = self
            .file
            .into_inner()
            .map_err(|e| anyhow::anyhow!("flush postings spill writer: {}", e.error()))?;
        // Patch the record_count at offset 8.
        f.seek(SeekFrom::Start(8))?;
        f.write_u64::<LittleEndian>(n)?;
        f.sync_all()?;
        Ok(n)
    }
}

/// Sequential reader over a `.postings.spill`. Yields one `PostingRecord` per
/// `next()` call.
#[derive(Debug)]
pub struct PostingsSpillReader {
    file: BufReader<File>,
    count_remaining: u64,
}

impl PostingsSpillReader {
    /// `buf_size` is the `BufReader` capacity. Postings are read in 16-byte
    /// chunks; the std default (8 KiB) issues a syscall every ~512 records.
    /// See `StreamingConfig::write_buffer_bytes` for the rationale.
    pub fn open(path: &Path, buf_size: usize) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut r = BufReader::with_capacity(buf_size, f);
        let magic = r.read_u32::<LittleEndian>()?;
        if magic != POSTINGS_SPILL_MAGIC {
            anyhow::bail!("bad postings spill magic: 0x{:08x}", magic);
        }
        let version = r.read_u32::<LittleEndian>()?;
        if version != SPILL_VERSION {
            anyhow::bail!("unsupported postings spill version: {}", version);
        }
        let count = r.read_u64::<LittleEndian>()?;
        Ok(Self {
            file: r,
            count_remaining: count,
        })
    }

    #[allow(dead_code)] // public reader-side helper for diagnostics / progress.
    pub fn records_remaining(&self) -> u64 {
        self.count_remaining
    }
}

impl Iterator for PostingsSpillReader {
    type Item = Result<PostingRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count_remaining == 0 {
            return None;
        }
        let tri_hash = match self.file.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => return Some(Err(e.into())),
        };
        let doc_id = match self.file.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => return Some(Err(e.into())),
        };
        let line_no = match self.file.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => return Some(Err(e.into())),
        };
        let byte_offset = match self.file.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => return Some(Err(e.into())),
        };
        self.count_remaining -= 1;
        Some(Ok(PostingRecord {
            tri_hash,
            doc_id,
            line_no,
            byte_offset,
        }))
    }
}

// ─── MasksSpill ───────────────────────────────────────────────────────────────

/// One (tri_hash, mask) entry in a `.masks.spill`. `tri_hash` is in the file
/// so the k-way merge can group by it without having to recover the original
/// trigram bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSpillEntry {
    pub tri_hash: u32,
    pub mask: FollowerMask,
}

impl MaskSpillEntry {
    #[allow(dead_code)] // public wire-format constant for callers wanting to size buffers.
    pub const SIZE: usize = 4 + 32;
}

/// Writer over a `.masks.spill` file. Caller sorts entries by `tri_hash`
/// ascending and pushes them; the writer emits a header + the concatenated
/// 36-byte entries.
pub struct MasksSpillWriter {
    file: BufWriter<File>,
}

impl MasksSpillWriter {
    /// `buf_size` is the `BufWriter` capacity for this spill. See
    /// `PostingsSpillWriter::create` for the rationale.
    pub fn create(path: &Path, buf_size: usize) -> Result<Self> {
        let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = BufWriter::with_capacity(buf_size, f);
        w.write_u32::<LittleEndian>(MASKS_SPILL_MAGIC)?;
        w.write_u32::<LittleEndian>(SPILL_VERSION)?;
        Ok(Self { file: w })
    }

    pub fn push(&mut self, e: &MaskSpillEntry) -> Result<()> {
        self.file.write_u32::<LittleEndian>(e.tri_hash)?;
        for word in e.mask {
            self.file.write_u32::<LittleEndian>(word)?;
        }
        Ok(())
    }

    #[allow(dead_code)] // public batch helper for callers emitting many entries.
    pub fn push_slice(&mut self, entries: &[MaskSpillEntry]) -> Result<()> {
        for e in entries {
            self.push(e)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.file.flush()?;
        let f = self
            .file
            .into_inner()
            .map_err(|e| anyhow::anyhow!("flush masks spill writer: {}", e.error()))?;
        f.sync_all()?;
        Ok(())
    }
}

/// Sequential reader over a `.masks.spill`. Yields one `MaskSpillEntry` per
/// `next()` call.
#[derive(Debug)]
pub struct MasksSpillReader {
    file: BufReader<File>,
    exhausted: bool,
}

impl MasksSpillReader {
    /// `buf_size` is the `BufReader` capacity. See
    /// `PostingsSpillReader::open` for the rationale.
    pub fn open(path: &Path, buf_size: usize) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut r = BufReader::with_capacity(buf_size, f);
        let magic = r.read_u32::<LittleEndian>()?;
        if magic != MASKS_SPILL_MAGIC {
            anyhow::bail!("bad masks spill magic: 0x{:08x}", magic);
        }
        let version = r.read_u32::<LittleEndian>()?;
        if version != SPILL_VERSION {
            anyhow::bail!("unsupported masks spill version: {}", version);
        }
        Ok(Self {
            file: r,
            exhausted: false,
        })
    }
}

impl Iterator for MasksSpillReader {
    type Item = Result<MaskSpillEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let tri_hash = match self.file.read_u32::<LittleEndian>() {
            Ok(v) => v,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    self.exhausted = true;
                    return None;
                }
                return Some(Err(e.into()));
            }
        };
        let mut mask = [0u32; 8];
        let mut read_ok = true;
        for word in mask.iter_mut() {
            match self.file.read_u32::<LittleEndian>() {
                Ok(v) => *word = v,
                Err(e) => {
                    read_ok = false;
                    if e.kind() != std::io::ErrorKind::UnexpectedEof {
                        return Some(Err(e.into()));
                    }
                }
            }
        }
        if !read_ok {
            self.exhausted = true;
            None
        } else {
            Some(Ok(MaskSpillEntry { tri_hash, mask }))
        }
    }
}

// ─── File naming helpers ─────────────────────────────────────────────────────

/// Naming convention for the spill files inside a `SpillDir`. `seq` is the
/// chunk index and distinguishes CS from CI: CS uses `.postings.spill` /
/// `.masks.spill`, CI uses `.ci.postings.spill` / `.ci.masks.spill`.
pub mod naming {
    use super::PathBuf;

    pub fn postings_path(dir: &super::Path, seq: u32, ci: bool) -> PathBuf {
        dir.join(format!(
            "{:08}.{}.postings.spill",
            seq,
            if ci { "ci" } else { "cs" }
        ))
    }

    pub fn masks_path(dir: &super::Path, seq: u32, ci: bool) -> PathBuf {
        dir.join(format!(
            "{:08}.{}.masks.spill",
            seq,
            if ci { "ci" } else { "cs" }
        ))
    }
}

// ─── Sort + spill helpers ─────────────────────────────────────────────────────

/// Sort `out.postings` (and `out.postings_ci` if non-empty) ascending by
/// `(tri_hash, doc_id, line_no, byte_offset)`. Then sort
/// `out.masks`/`out.masks_ci` (HashMaps) into `Vec<(tri, FollowerMask)>`
/// ordered by `crc32fast::hash(tri)` ascending — same order the legacy
/// `write_ngram_files` emits (`src/persist.rs:965`).
pub fn sort_chunk_outputs(out: &mut ChunkOutputs) {
    sort_postings(&mut out.postings);
    sort_postings(&mut out.postings_ci);
}

fn sort_postings(recs: &mut [PostingRecord]) {
    if recs.is_empty() {
        return;
    }
    recs.sort_unstable_by_key(|r| (r.tri_hash, r.doc_id, r.line_no, r.byte_offset));
}

/// Collect sorted `(tri, mask)` pairs from a chunk HashMap into a `Vec` (and a
/// parallel hash list). Order: `crc32fast::hash(tri)` ascending, matching the
/// legacy `write_ngram_files` ordering at `src/persist.rs:965`.
#[allow(dead_code)] // helper retained for callers / tests; the spill path uses `write_masks_spill` directly.
pub fn collect_sorted_masks(
    masks: &HashMap<[u8; 3], FollowerMask>,
) -> Vec<(u32, [u8; 3], FollowerMask)> {
    let mut out: Vec<(u32, [u8; 3], FollowerMask)> = masks
        .iter()
        .map(|(tri, mask)| (crc32fast::hash(tri), *tri, *mask))
        .collect();
    out.sort_unstable_by_key(|(h, _, _)| *h);
    out
}

// ─── StreamingConfig (M2/M3) ─────────────────────────────────────────────────

/// Default buffer size for every `BufWriter` / `BufReader` involved in the
/// streaming build (spill files + final index files + spill readers). The
/// `std::io` default of 8 KiB is far too small for this workload — postings
/// are written in 16-byte records and read in the same size, so the default
/// fires a syscall every ~512 records. 1 MiB cuts syscalls ~128x and is the
/// dominant lever for build throughput on spinning disks and Windows (where
/// `write()` syscall overhead is much higher than on Linux).
pub const DEFAULT_WRITE_BUFFER_BYTES: usize = 1024 * 1024;

/// Knobs for the streaming build path. Defaults balance peak RAM
/// (chunk working set + per-trigram merge buffer) against build throughput.
/// The defaults target a 16-64 GB machine processing an AOSP-sized corpus.
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Maximum number of files per chunk. Each chunk's working set (file
    /// bytes + extracted postings + masks hashmap) fits in this budget.
    pub chunk_files: usize,

    /// Soft cap on accumulated raw file bytes per chunk. A chunk is spilled
    /// when either `chunk_files` OR `chunk_byte_target` is reached first.
    pub chunk_byte_target: usize,

    /// Per-trigram posting buffer cap during the merge pass. When a
    /// trigram's encoded postings would exceed this, a `tracing::warn!` is
    /// emitted and accumulation continues (v1 ships without per-trigram
    /// side-spilling; see plan §8.1).
    pub max_posting_buf_bytes: usize,

    /// Buffer size for every `BufWriter` / `BufReader` involved in the
    /// streaming build (spill files, final index files, spill readers).
    /// See `DEFAULT_WRITE_BUFFER_BYTES` for the rationale on the default.
    pub write_buffer_bytes: usize,

    /// When true (default), the first chunk is accumulated in-RAM using the
    /// same `HashMap`-based path the legacy `add_document` uses; if at the
    /// end only one chunk was produced, the spill-and-merge phase is skipped
    /// entirely. Yields zero overhead for small corpora.
    #[allow(dead_code)]
    // accepted as a config knob for future use; current impl always spills + merges.
    pub defer_first_chunk: bool,

    /// Print progress messages during extraction and merge to stderr.
    pub verbose: bool,
}

impl StreamingConfig {
    pub fn defaults() -> Self {
        Self {
            chunk_files: 4096,
            chunk_byte_target: 512 * 1024 * 1024,     // 512 MiB
            max_posting_buf_bytes: 512 * 1024 * 1024, // 512 MiB
            write_buffer_bytes: DEFAULT_WRITE_BUFFER_BYTES,
            defer_first_chunk: true,
            verbose: false,
        }
    }

    /// Same as `defaults()` but with `verbose` set, for callers that already
    /// have a `bool` rather than rebuilding the struct field-by-field.
    pub fn defaults_with_verbose(verbose: bool) -> Self {
        Self {
            verbose,
            ..Self::defaults()
        }
    }
}

// ─── K-way merge (M2) ────────────────────────────────────────────────────────

/// Result of merging one store (CS or CI). Used by callers to populate
/// `IndexMeta` and to drive verbose progress.
#[derive(Clone, Copy, Debug, Default)]
pub struct MergeResult {
    pub num_trigrams: usize,
    pub postings_len: u64,
}

/// Wrap a `PostingsSpillReader` for k-way merge: expose the smallest record
/// (`cur`) plus the open iterator; advance consumes from the iterator and
/// updates `cur`.
struct PostingsFront {
    iter: PostingsSpillReader,
    cur: Option<PostingRecord>,
}

impl PostingsFront {
    fn open(path: &Path, buf_size: usize) -> Result<Self> {
        let mut iter = PostingsSpillReader::open(path, buf_size)?;
        let cur = match iter.next() {
            Some(Ok(r)) => Some(r),
            Some(Err(e)) => return Err(e),
            None => None,
        };
        Ok(Self { iter, cur })
    }

    fn advance(&mut self) {
        match self.iter.next() {
            Some(Ok(r)) => self.cur = Some(r),
            _ => self.cur = None,
        }
    }
}

/// Wrap a `MasksSpillReader` the same way.
struct MasksFront {
    iter: MasksSpillReader,
    cur: Option<MaskSpillEntry>,
}

impl MasksFront {
    fn open(path: &Path, buf_size: usize) -> Result<Self> {
        let mut iter = MasksSpillReader::open(path, buf_size)?;
        let cur = match iter.next() {
            Some(Ok(e)) => Some(e),
            Some(Err(e)) => return Err(e),
            None => None,
        };
        Ok(Self { iter, cur })
    }

    fn advance(&mut self) {
        match self.iter.next() {
            Some(Ok(e)) => self.cur = Some(e),
            _ => self.cur = None,
        }
    }
}

/// Min-heap entry for the postings merge. Ordering on `(tri_hash, doc_id,
/// line_no, byte_offset, spill_idx)` gives the same order the original sorted
/// union would yield; `spill_idx` only acts as a tiebreaker.
struct PostingsHeapItem {
    tri_hash: u32,
    doc_id: u32,
    line_no: u32,
    byte_offset: u32,
    spill_idx: u32,
}

impl PartialEq for PostingsHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.tri_hash == other.tri_hash
            && self.doc_id == other.doc_id
            && self.line_no == other.line_no
            && self.byte_offset == other.byte_offset
            && self.spill_idx == other.spill_idx
    }
}
impl Eq for PostingsHeapItem {}
impl Ord for PostingsHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tri_hash
            .cmp(&other.tri_hash)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
            .then_with(|| self.line_no.cmp(&other.line_no))
            .then_with(|| self.byte_offset.cmp(&other.byte_offset))
            .then_with(|| self.spill_idx.cmp(&other.spill_idx))
    }
}
impl PartialOrd for PostingsHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap entry for the masks merge. Ordered by `(tri_hash, spill_idx)`.
struct MasksHeapItem {
    tri_hash: u32,
    spill_idx: u32,
}

impl PartialEq for MasksHeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.tri_hash == other.tri_hash && self.spill_idx == other.spill_idx
    }
}
impl Eq for MasksHeapItem {}
impl Ord for MasksHeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.tri_hash
            .cmp(&other.tri_hash)
            .then_with(|| self.spill_idx.cmp(&other.spill_idx))
    }
}
impl PartialOrd for MasksHeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Run the k-way merge for one store (CS or CI). Opens the six output files
/// under `output_dir` with the given `prefix`, walks all postings spill
/// fronts in `(tri_hash, doc_id, line_no, byte_offset)` ascending order while
/// OR-accumulating masks for each trigram, and writes
/// `{prefix}.postings`+`{prefix}.lookup`, `{prefix}.bitmaps`+`{prefix}.bitmaps.lookup`,
/// `{prefix}.masks`+`{prefix}.masks.lookup` in `crc32fast::hash(tri)` order
/// across all three (matching `persist::write_ngram_files` at
/// `src/persist.rs:957-1055`).
pub fn merge_to_store(
    output_dir: &Path,
    prefix: &str,
    postings_spills: &[PathBuf],
    masks_spills: &[PathBuf],
    cfg: &StreamingConfig,
) -> Result<MergeResult> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let postings_path = output_dir.join(format!("{prefix}.postings"));
    let postings_lookup_path = output_dir.join(format!("{prefix}.lookup"));
    let bitmaps_path = output_dir.join(format!("{prefix}.bitmaps"));
    let bitmaps_lookup_path = output_dir.join(format!("{prefix}.bitmaps.lookup"));
    let masks_path = output_dir.join(format!("{prefix}.masks"));
    let masks_lookup_path = output_dir.join(format!("{prefix}.masks.lookup"));

    // Single buffer size for every output BufWriter here. Postings is by far
    // the largest output and dominates throughput, so using the same size
    // everywhere keeps the merge flush cadence predictable. See
    // `StreamingConfig::write_buffer_bytes` for the default and rationale.
    let buf_size = cfg.write_buffer_bytes;

    let mut postings_file = BufWriter::with_capacity(
        buf_size,
        File::create(&postings_path)
            .with_context(|| format!("creating {}", postings_path.display()))?,
    );
    let mut postings_lookup_file = BufWriter::with_capacity(
        buf_size,
        File::create(&postings_lookup_path)
            .with_context(|| format!("creating {}", postings_lookup_path.display()))?,
    );
    let mut bitmaps_file = BufWriter::with_capacity(
        buf_size,
        File::create(&bitmaps_path)
            .with_context(|| format!("creating {}", bitmaps_path.display()))?,
    );
    let mut bitmaps_lookup_file = BufWriter::with_capacity(
        buf_size,
        File::create(&bitmaps_lookup_path)
            .with_context(|| format!("creating {}", bitmaps_lookup_path.display()))?,
    );
    let mut masks_file = BufWriter::with_capacity(
        buf_size,
        File::create(&masks_path).with_context(|| format!("creating {}", masks_path.display()))?,
    );
    let mut masks_lookup_file = BufWriter::with_capacity(
        buf_size,
        File::create(&masks_lookup_path)
            .with_context(|| format!("creating {}", masks_lookup_path.display()))?,
    );

    // Open all fronts and prime the heaps.
    let mut postings_fronts: Vec<PostingsFront> = Vec::with_capacity(postings_spills.len());
    for p in postings_spills {
        postings_fronts.push(PostingsFront::open(p, buf_size)?);
    }
    let mut masks_fronts: Vec<MasksFront> = Vec::with_capacity(masks_spills.len());
    for p in masks_spills {
        masks_fronts.push(MasksFront::open(p, buf_size)?);
    }

    let mut postings_heap: BinaryHeap<Reverse<PostingsHeapItem>> =
        BinaryHeap::with_capacity(postings_fronts.len());
    for (i, front) in postings_fronts.iter_mut().enumerate() {
        if let Some(rec) = front.cur {
            postings_heap.push(Reverse(PostingsHeapItem {
                tri_hash: rec.tri_hash,
                doc_id: rec.doc_id,
                line_no: rec.line_no,
                byte_offset: rec.byte_offset,
                spill_idx: i as u32,
            }));
        }
    }

    let mut masks_heap: BinaryHeap<Reverse<MasksHeapItem>> =
        BinaryHeap::with_capacity(masks_fronts.len());
    for (i, front) in masks_fronts.iter_mut().enumerate() {
        if let Some(entry) = front.cur {
            masks_heap.push(Reverse(MasksHeapItem {
                tri_hash: entry.tri_hash,
                spill_idx: i as u32,
            }));
        }
    }

    // Per-trigram accumulators.
    let mut cur_writer = crate::postenc::PostingWriter::new();
    let mut cur_posting_buf: Vec<u8> = Vec::new();
    let mut cur_bitmap = roaring::RoaringBitmap::new();
    let mut cur_mask: FollowerMask = [0u32; 8];
    let mut cur_tri_hash: Option<u32> = None;
    let mut warned_cap = false;

    let mut postings_offset: u64 = 0;
    let mut bitmaps_offset: u64 = 0;
    let mut masks_offset: u64 = 0;
    let mut num_trigrams: usize = 0;

    /// Flush the current trigram (if any) to the output files. Resets the
    /// per-trigram accumulators and bumps the offsets/count.
    fn flush_trigram(
        tri_hash: u32,
        posting_buf: &[u8],
        bitmap: &mut roaring::RoaringBitmap,
        mask: &FollowerMask,
        postings_file: &mut BufWriter<File>,
        postings_lookup_file: &mut BufWriter<File>,
        bitmaps_file: &mut BufWriter<File>,
        bitmaps_lookup_file: &mut BufWriter<File>,
        masks_file: &mut BufWriter<File>,
        masks_lookup_file: &mut BufWriter<File>,
        postings_offset: &mut u64,
        bitmaps_offset: &mut u64,
        masks_offset: &mut u64,
    ) -> Result<()> {
        let post_len = posting_buf.len() as u32;
        postings_file.write_all(posting_buf)?;
        postings_lookup_file.write_u32::<LittleEndian>(tri_hash)?;
        postings_lookup_file.write_u64::<LittleEndian>(*postings_offset)?;
        postings_lookup_file.write_u32::<LittleEndian>(post_len)?;
        *postings_offset += post_len as u64;

        let mut bm_bytes = Vec::with_capacity(64);
        bitmap.serialize_into(&mut bm_bytes)?;
        let bm_len = bm_bytes.len() as u32;
        bitmaps_file.write_all(&bm_bytes)?;
        bitmaps_lookup_file.write_u32::<LittleEndian>(tri_hash)?;
        bitmaps_lookup_file.write_u64::<LittleEndian>(*bitmaps_offset)?;
        bitmaps_lookup_file.write_u32::<LittleEndian>(bm_len)?;
        *bitmaps_offset += bm_len as u64;

        for word in mask {
            masks_file.write_u32::<LittleEndian>(*word)?;
        }
        masks_lookup_file.write_u32::<LittleEndian>(tri_hash)?;
        masks_lookup_file.write_u64::<LittleEndian>(*masks_offset)?;
        masks_lookup_file.write_u32::<LittleEndian>(MASK_ENTRY_SIZE_CONST as u32)?;
        *masks_offset += MASK_ENTRY_SIZE_CONST as u64;

        bitmap.clear();
        Ok(())
    }

    // Helper closure that drains the postings heap for one tri_hash into the
    // cur_* accumulators (with `last_dl` dedup).

    // Main loop: walk postings in tri_hash order.
    let mut next_merge_progress: u64 = 100_000;
    while let Some(top_hash) = postings_heap.peek().map(|Reverse(item)| item.tri_hash) {
        let tri_hash = top_hash;

        // If advancing to a new trigram, flush the previous one.
        if let Some(prev) = cur_tri_hash {
            if prev != tri_hash {
                flush_trigram(
                    prev,
                    &cur_posting_buf,
                    &mut cur_bitmap,
                    &cur_mask,
                    &mut postings_file,
                    &mut postings_lookup_file,
                    &mut bitmaps_file,
                    &mut bitmaps_lookup_file,
                    &mut masks_file,
                    &mut masks_lookup_file,
                    &mut postings_offset,
                    &mut bitmaps_offset,
                    &mut masks_offset,
                )?;
                num_trigrams += 1;
                cur_posting_buf.clear();
                cur_writer = crate::postenc::PostingWriter::new();
                // cur_bitmap was cleared inside flush_trigram
                cur_mask = [0u32; 8];

                if cfg.verbose && num_trigrams as u64 >= next_merge_progress {
                    eprintln!(
                        "  merging: {} trigrams, {} MB written",
                        num_trigrams,
                        postings_offset / (1024 * 1024)
                    );
                    next_merge_progress = num_trigrams as u64 + 100_000;
                }
            }
        }

        cur_tri_hash = Some(tri_hash);

        // Drain all postings for this tri_hash.
        loop {
            let still_this = match postings_heap.peek() {
                Some(Reverse(it)) => it.tri_hash == tri_hash,
                None => false,
            };
            if !still_this {
                break;
            }
            let Reverse(item) = postings_heap.pop().unwrap();
            let spill_idx = item.spill_idx as usize;
            let front = &mut postings_fronts[spill_idx];
            let rec = front
                .cur
                .expect("heap promised this front had a cur record");
            front.advance();
            if let Some(next) = front.cur {
                postings_heap.push(Reverse(PostingsHeapItem {
                    tri_hash: next.tri_hash,
                    doc_id: next.doc_id,
                    line_no: next.line_no,
                    byte_offset: next.byte_offset,
                    spill_idx: spill_idx as u32,
                }));
            }
            if cur_writer.last_dl() != Some((rec.doc_id, rec.line_no)) {
                cur_writer.push(
                    &mut cur_posting_buf,
                    rec.doc_id,
                    rec.line_no,
                    rec.byte_offset,
                );
                cur_bitmap.insert(rec.doc_id);
            }
        }

        // Drain masks with tri_hash <= tri_hash into cur_mask (only == contribute;
        // < already had no postings so we drop them).
        loop {
            let next_mask = match masks_heap.peek() {
                Some(Reverse(it)) => it.tri_hash,
                None => break,
            };
            if next_mask > tri_hash {
                break;
            }
            let Reverse(item) = masks_heap.pop().unwrap();
            let spill_idx = item.spill_idx as usize;
            let front = &mut masks_fronts[spill_idx];
            let entry = front.cur.expect("heap promised this front had a cur entry");
            front.advance();
            if let Some(next) = front.cur {
                masks_heap.push(Reverse(MasksHeapItem {
                    tri_hash: next.tri_hash,
                    spill_idx: spill_idx as u32,
                }));
            }
            if entry.tri_hash == tri_hash {
                for (i, w) in entry.mask.iter().enumerate() {
                    cur_mask[i] |= *w;
                }
            }
        }

        if cur_posting_buf.len() > cfg.max_posting_buf_bytes && !warned_cap {
            warned_cap = true;
            eprintln!(
                "[fgr] warning: trigram 0x{:08x} exceeded per-trigram buffer cap ({} bytes); v1 keeps accumulating — see plan §8.1",
                tri_hash, cfg.max_posting_buf_bytes
            );
        }
    }

    // Final flush.
    if let Some(prev) = cur_tri_hash {
        flush_trigram(
            prev,
            &cur_posting_buf,
            &mut cur_bitmap,
            &cur_mask,
            &mut postings_file,
            &mut postings_lookup_file,
            &mut bitmaps_file,
            &mut bitmaps_lookup_file,
            &mut masks_file,
            &mut masks_lookup_file,
            &mut postings_offset,
            &mut bitmaps_offset,
            &mut masks_offset,
        )?;
        num_trigrams += 1;
    }

    if cfg.verbose {
        eprintln!(
            "  merging done: {} trigrams, {} MB written",
            num_trigrams,
            postings_offset / (1024 * 1024)
        );
    }

    // Close all writers (drop them via binding end-of-scope after flush).
    postings_file.flush()?;
    postings_lookup_file.flush()?;
    bitmaps_file.flush()?;
    bitmaps_lookup_file.flush()?;
    masks_file.flush()?;
    masks_lookup_file.flush()?;

    Ok(MergeResult {
        num_trigrams,
        postings_len: postings_offset,
    })
}

// 32 bytes / mask entry. Duplicated from the on-disk format (the legacy
// writer in `src/persist.rs:957` does the same).
const MASK_ENTRY_SIZE_CONST: usize = 32;

// ─── Streaming glue (M3) ──────────────────────────────────────────────────────

/// Result of a full streaming build. Returned to `persist::build` /
/// `persist::compact` so they can populate `IndexMeta`.
#[derive(Clone, Debug, Default)]
pub struct BuildResult {
    /// Number of documents actually ingested (binary skips and read errors are
    /// silently dropped, matching the legacy `build_from_paths` semantics).
    pub num_docs: u32,
    /// Paths actually accepted and indexed, in compact-doc-id order.
    /// `indexed_paths[i]` corresponds to `doc_id = i`.
    pub indexed_paths: Vec<PathBuf>,
    /// Unique trigram count in the case-sensitive store.
    pub num_ngrams: usize,
    /// Bytes written to `ngrams.postings` (CS).
    pub postings_len: u64,
}

/// State that progresses through chunks during a streaming build. Owns the
/// spill directory; the k-way merge runs once at `finish()` time, so each
/// chunk's working set is released before the next starts. The spill temp
/// directory is RAII-cleaned via `SpillDir`'s `Drop` impl.
pub struct StreamingBuilder {
    spill_dir: SpillDir,
    current: ChunkOutputs,
    current_files: usize,
    current_bytes: u64,
    case_insensitive: bool,
    cfg: StreamingConfig,
    seq: u32,
    postings_spills: Vec<PathBuf>,
    masks_spills: Vec<PathBuf>,
    postings_spills_ci: Vec<PathBuf>,
    masks_spills_ci: Vec<PathBuf>,
}

impl StreamingBuilder {
    /// Allocate scratch dir and prepare accumulator state.
    pub fn new(case_insensitive: bool, cfg: StreamingConfig) -> Result<Self> {
        Ok(Self {
            spill_dir: SpillDir::new()?,
            current: ChunkOutputs::new(),
            current_files: 0,
            current_bytes: 0,
            case_insensitive,
            cfg,
            seq: 0,
            postings_spills: Vec::new(),
            masks_spills: Vec::new(),
            postings_spills_ci: Vec::new(),
            masks_spills_ci: Vec::new(),
        })
    }

    /// Add one document's worth of postings + masks to the in-flight chunk.
    /// Triggers a chunk spill (sort + write to disk + reset) when either the
    /// file-count or the file-byte threshold is exceeded.
    pub fn process_document(&mut self, doc_id: u32, content: &[u8]) -> Result<()> {
        extract_file(content, doc_id, self.case_insensitive, &mut self.current);
        self.current_files += 1;
        self.current_bytes += content.len() as u64;

        let full = self.current_files >= self.cfg.chunk_files
            || self.current_bytes as usize >= self.cfg.chunk_byte_target;
        if full {
            self.spill_current()?;
        }
        Ok(())
    }

    /// Sort + spill the in-flight chunk to disk. No-op when the chunk is
    /// empty.
    fn spill_current(&mut self) -> Result<()> {
        if self.current_files == 0 && self.current.postings.is_empty() {
            return Ok(());
        }

        let seq = self.seq;
        self.seq += 1;

        if self.cfg.verbose {
            eprintln!(
                "  spill chunk {} ({} files, {} bytes, {} cs postings)",
                seq,
                self.current_files,
                self.current_bytes,
                self.current.postings.len(),
            );
        }

        sort_chunk_outputs(&mut self.current);

        // Plumbed into every spill writer so the per-chunk write path uses
        // the same large BufWriter buffer as the k-way merge output.
        let buf_size = self.cfg.write_buffer_bytes;

        // CS postings spill.
        let p_path = naming::postings_path(self.spill_dir.path(), seq, false);
        {
            let mut w = PostingsSpillWriter::create(&p_path, buf_size)?;
            for &rec in &self.current.postings {
                w.push(rec)?;
            }
            w.finish()?;
        }
        self.postings_spills.push(p_path);

        // CS masks spill.
        let m_path = naming::masks_path(self.spill_dir.path(), seq, false);
        write_masks_spill(&m_path, &self.current.masks, buf_size)?;
        self.masks_spills.push(m_path);

        // CI companion.
        if self.case_insensitive {
            let pci_path = naming::postings_path(self.spill_dir.path(), seq, true);
            {
                let mut w = PostingsSpillWriter::create(&pci_path, buf_size)?;
                for &rec in &self.current.postings_ci {
                    w.push(rec)?;
                }
                w.finish()?;
            }
            self.postings_spills_ci.push(pci_path);

            let mci_path = naming::masks_path(self.spill_dir.path(), seq, true);
            write_masks_spill(&mci_path, &self.current.masks_ci, buf_size)?;
            self.masks_spills_ci.push(mci_path);
        }

        self.current.clear();
        self.current_files = 0;
        self.current_bytes = 0;
        Ok(())
    }

    /// Spill the final (possibly partial) chunk, then run k-way merge of all
    /// spills into `output_dir/{prefix}.*` for the CS index, then again for
    /// the CI companion when applicable. The spill temp dir is dropped on
    /// return, removing all temp files.
    pub fn finish(mut self, output_dir: &Path) -> Result<MergeResult> {
        self.spill_current()?;

        let cs = merge_to_store(
            output_dir,
            "ngrams",
            &self.postings_spills,
            &self.masks_spills,
            &self.cfg,
        )?;

        if self.case_insensitive {
            let _ci = merge_to_store(
                output_dir,
                "ngrams.ci",
                &self.postings_spills_ci,
                &self.masks_spills_ci,
                &self.cfg,
            )?;
        }

        Ok(cs)
    }
}

/// Serialize `masks` (HashMap) to a spill file in `crc32fast::hash(tri)`
/// ascending order. Trigrams with no recorded followers still get a zero
/// mask entry so the binary search on the merged mask file stays aligned.
fn write_masks_spill(
    path: &Path,
    masks: &HashMap<[u8; 3], FollowerMask>,
    buf_size: usize,
) -> Result<()> {
    let mut sorted: Vec<([u8; 3], FollowerMask)> = masks.iter().map(|(k, v)| (*k, *v)).collect();
    sorted.sort_by_key(|(k, _)| crc32fast::hash(k));
    let mut w = MasksSpillWriter::create(path, buf_size)?;
    for (tri, mask) in &sorted {
        let tri_hash = crc32fast::hash(tri);
        w.push(&MaskSpillEntry {
            tri_hash,
            mask: *mask,
        })?;
    }
    w.finish()?;
    Ok(())
}

/// Build the streaming index directly from a pre-collected path list. Used by
/// both `streaming_build` (which walks the directory itself) and by
/// `persist::compact` (which already knows which files belong to the corpus
/// from the persisted doc-ids).
///
/// Files unreadable or detected as binary are silently skipped — the legacy
/// `SparseIndex::build_from_paths` does the same, so callers see identical
/// `num_docs` semantics.
pub fn streaming_build_from_paths(
    paths: &[PathBuf],
    case_insensitive: bool,
    cfg: &StreamingConfig,
    output_dir: &Path,
) -> Result<BuildResult> {
    use rayon::prelude::*;

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;

    // Single-chunk fast path: when `defer_first_chunk` is enabled and the
    // corpus is small enough to fit in one chunk (file count and byte count
    // both under their caps), use the legacy in-RAM `SparseIndex` builder
    // and serialize directly. Skips the spill-to-disk + k-way merge entirely,
    // recovering the ~2x build-time regression that small corpora saw when
    // the streaming build was introduced. On multi-chunk corpora the
    // streaming path below remains in charge.
    if cfg.defer_first_chunk
        && paths.len() <= cfg.chunk_files
        && corpus_total_bytes(paths) <= cfg.chunk_byte_target as u64
    {
        return build_legacy_fast_path(paths, case_insensitive, cfg.verbose, output_dir);
    }

    let mut builder = StreamingBuilder::new(case_insensitive, cfg.clone())?;
    let chunk_files = builder.cfg.chunk_files;
    let verbose = builder.cfg.verbose;

    let mut count = 0u32;
    let mut indexed_paths: Vec<PathBuf> = Vec::new();
    let mut next_progress = 10_000u32;

    for chunk_start in (0..paths.len()).step_by(chunk_files) {
        let chunk_end = (chunk_start + chunk_files).min(paths.len());
        let chunk = &paths[chunk_start..chunk_end];

        let contents: Vec<Option<Vec<u8>>> = chunk
            .par_iter()
            .map(|path| -> Option<Vec<u8>> {
                let content = std::fs::read(path).ok()?;
                if !crate::searcher::is_known_text_ext(path)
                    && content.iter().take(512).any(|&b| b == 0)
                {
                    return None;
                }
                Some(content)
            })
            .collect();

        for (path, content) in chunk.iter().zip(contents.into_iter()) {
            if let Some(content) = content {
                builder.process_document(count, &content)?;
                indexed_paths.push(path.to_path_buf());
                count += 1;
                if verbose && count >= next_progress {
                    eprintln!("  indexed {} files...", count);
                    next_progress = count + 10_000;
                }
            }
        }
    }

    let cs = builder.finish(output_dir)?;
    debug_assert_eq!(indexed_paths.len(), count as usize);
    Ok(BuildResult {
        num_docs: count,
        indexed_paths,
        num_ngrams: cs.num_trigrams,
        postings_len: cs.postings_len,
    })
}

/// Sum of file sizes across `paths`, ignoring entries that fail `metadata`
/// (e.g. dangling symlinks). Used by the single-chunk fast-path gate; an
/// approximate sum is fine because the chunk-byte cap is a soft target.
fn corpus_total_bytes(paths: &[PathBuf]) -> u64 {
    let mut total: u64 = 0;
    for p in paths {
        if let Ok(m) = std::fs::metadata(p) {
            total = total.saturating_add(m.len());
        }
    }
    total
}

/// Single-chunk legacy fast path: builds a `SparseIndex` in RAM and writes
/// the six output files directly via `persist::write_ngram_files`. Matches
/// the legacy `SparseIndex::add_document` semantics (binary-skip, dedup
/// per `(tri, doc, line)`) and produces the same on-disk format as the
/// streaming k-way merge, so an index built via this path is
/// indistinguishable from one built via the streaming path.
fn build_legacy_fast_path(
    paths: &[PathBuf],
    case_insensitive: bool,
    verbose: bool,
    output_dir: &Path,
) -> Result<BuildResult> {
    use crate::index::SparseIndex;

    if verbose {
        eprintln!(
            "  fast path: {} files fit in one chunk — using legacy in-RAM builder",
            paths.len()
        );
    }

    let mut index = SparseIndex::build_from_paths(paths, case_insensitive, verbose)?;
    let num_docs = index.doc_ids.len() as u32;
    let num_ngrams = index.ngrams.len();

    let postings_len =
        crate::persist::write_ngram_files(output_dir, "ngrams", &index.ngrams, &index.masks)?;

    if let Some(ref ci) = index.ngrams_ci {
        // The CI mask is paired with the CI trigram map; `add_document`
        // builds them lockstep. The empty-map fallback mirrors the legacy
        // `write_index_files` defensive path.
        static EMPTY: std::sync::OnceLock<HashMap<[u8; 3], crate::index::FollowerMask>> =
            std::sync::OnceLock::new();
        let ci_masks = index
            .masks_ci
            .as_ref()
            .unwrap_or_else(|| EMPTY.get_or_init(HashMap::new));
        crate::persist::write_ngram_files(output_dir, "ngrams.ci", ci, ci_masks)?;
    }

    Ok(BuildResult {
        num_docs,
        indexed_paths: std::mem::take(&mut index.doc_ids),
        num_ngrams,
        postings_len,
    })
}

/// Walk `root` with the project's git/ignore/type-filter rules, returning the
/// matching file paths in walker order. Callers (notably `persist::build`)
/// reuse this list to write `docids.bin` + capture per-file `mtime` for
/// `meta.json`.
/// Walk `root` and return the matching file paths, in walker order. Any
/// directory under `root` whose first component starts with `.fgr` is
/// excluded — these are the persistent-index output dirs left behind by
/// `fgr index` (default `--output .fgr`, but any `.fgr*` name is reserved
/// to be safe). Without this skip, a subsequent `fgr index` re-reads the
/// prior run's `ngrams.postings` — a varint-only stream whose first 512
/// bytes have no NUL, so the binary check misclassifies it as text — and
/// the resulting k-way merge inflates the new index by ~200x.
#[allow(dead_code)] // public API; `persist::build` uses it.
pub fn collect_paths(
    root: &Path,
    output_dir: &Path,
    no_ignore: bool,
    type_filter: &[String],
) -> Result<Vec<PathBuf>> {
    use ignore::WalkBuilder;

    let root = root.to_path_buf();
    let excluded = output_dir.to_path_buf();
    let walker = WalkBuilder::new(&root)
        .git_ignore(!no_ignore)
        .hidden(false)
        .filter_entry(move |entry| {
            // Skip the configured output dir AND any sibling `.fgr*` dir
            // (older builds may have written to `.fgr-test`, `.fgr-stream`,
            // etc., and those would otherwise be re-ingested as text).
            if entry.path().starts_with(&excluded) {
                return false;
            }
            if let Ok(rel) = entry.path().strip_prefix(&root) {
                if let Some(first) = rel.components().next() {
                    if let std::path::Component::Normal(name) = first {
                        if let Some(name_str) = name.to_str() {
                            if name_str.starts_with(".fgr") {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        })
        .build();

    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in walker {
        let entry = entry?;
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        if !crate::searcher::passes_type_filter(path, type_filter) {
            continue;
        }
        paths.push(path.to_path_buf());
    }
    Ok(paths)
}

/// Walk `root` with the project's git/ignore/type-filter rules, then
/// stream-build the index into `output`. Returns the same `BuildResult` as
/// `streaming_build_from_paths` so callers can populate `IndexMeta`.
#[allow(dead_code)] // convenience entry; `persist::build` uses `collect_paths` + `_from_paths`.
pub fn streaming_build(
    root: &Path,
    output: &Path,
    no_ignore: bool,
    type_filter: &[String],
    case_insensitive: bool,
    cfg: &StreamingConfig,
) -> Result<BuildResult> {
    let paths = collect_paths(root, output, no_ignore, type_filter)?;
    if cfg.verbose {
        eprintln!("Streaming build: {} files matching filters", paths.len());
    }
    streaming_build_from_paths(&paths, case_insensitive, cfg, output)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Buffer size used by every spill writer / reader in the unit tests
    /// below. Matches `DEFAULT_WRITE_BUFFER_BYTES`; kept as a local alias so
    /// each call site reads naturally (`create(&p, TEST_BUF)`).
    const TEST_BUF: usize = DEFAULT_WRITE_BUFFER_BYTES;

    /// Helper: dedup adjacent equal `(tri_hash, doc, line)` after sort and
    /// return the resulting unique `(tri_hash, doc, line, off)` tuples. This
    /// mirrors what the k-way merge will do at the spill-boundary in M2.
    fn dedup_sorted_postings(recs: Vec<PostingRecord>) -> Vec<(u32, u32, u32, u32)> {
        let mut sorted = recs;
        sort_postings(&mut sorted);
        let mut out: Vec<(u32, u32, u32, u32)> = Vec::with_capacity(sorted.len());
        for r in sorted {
            let t = (r.tri_hash, r.doc_id, r.line_no, r.byte_offset);
            if out.last() != Some(&t) {
                out.push(t);
            }
        }
        out
    }

    fn mask_has(mask: &FollowerMask, b: u8) -> bool {
        mask[(b >> 5) as usize] & (1u32 << (b & 0x1F)) != 0
    }

    #[test]
    fn extract_basic_cs() {
        let mut out = ChunkOutputs::new();
        extract_file(b"hello world", 0, false, &mut out);

        // Posting records for the CS pass are non-empty.
        assert!(!out.postings.is_empty());
        // CI outputs untouched.
        assert!(out.postings_ci.is_empty());
        assert!(out.masks_ci.is_empty());
    }

    #[test]
    fn extract_ci_lockstep() {
        let mut out = ChunkOutputs::new();
        extract_file(b"Hello WORLD", 0, true, &mut out);

        // CS map contains original-case "Hel".
        assert!(out.masks.contains_key(b"Hel"));
        // CI map should contain folded "hel" (the lowercase equivalent).
        assert!(out.masks_ci.contains_key(b"hel"));
        assert!(!out.masks_ci.contains_key(b"Hel"));
    }

    #[test]
    fn extract_cross_line_follower_for_cs_mask() {
        // Regression for the cross-line-follower invariant (mirrors
        // `src/index.rs:411-420`).
        let mut out = ChunkOutputs::new();
        extract_file(b"abc\ndef", 0, false, &mut out);
        let m = out.masks.get(b"bc\n").expect("CS mask for bc\\n");
        assert!(mask_has(m, b'd'), "cross-line follower must be captured");
    }

    #[test]
    fn extract_short_content_skips_line_pass() {
        let mut out = ChunkOutputs::new();
        extract_file(b"ab", 0, false, &mut out);
        // < 3 bytes → no postings, no masks.
        assert!(out.postings.is_empty());
        assert!(out.masks.is_empty());
    }

    #[test]
    fn extract_lines_shorter_than_three_are_skipped() {
        // First line: "abc" (3 trigrams of length 3 — wait, only "abc" → 1
        // trigram). Second line: "x" (skipped). Third line: "abcdef" (4
        // trigrams).
        let mut out = ChunkOutputs::new();
        extract_file(b"abc\nx\nabcdef", 0, false, &mut out);
        let counts = dedup_sorted_postings(out.postings.clone());
        // We expect postings on doc 0, lines 1 ("abc" contributes 1 trigram)
        // and line 3 ("abcdef" contributes 4 trigrams).
        let lines: std::collections::HashSet<u32> = counts.iter().map(|(_, _, l, _)| *l).collect();
        assert!(lines.contains(&1));
        assert!(lines.contains(&3));
        assert!(!lines.contains(&2), "short line must be skipped");
    }

    #[test]
    fn dedup_preserves_one_posting_per_trigram_per_line() {
        // Same trigram ABC repeated 3 times in one line → one posting only.
        // Same trigram ABC in two different lines → two postings.
        let mut out = ChunkOutputs::new();
        extract_file(b"abcabcabc\nabc", 0, false, &mut out);
        let counts = dedup_sorted_postings(out.postings.clone());
        // Both occurrences of "abc" trigram should be exactly 2 unique: one
        // for line 1 and one for line 2. The 3 repeats within line 1 collapse
        // to one.
        let abc_hash = crc32fast::hash(b"abc");
        let abc_count = counts.iter().filter(|(h, _, _, _)| *h == abc_hash).count();
        assert_eq!(abc_count, 2, "expected 2 unique abc postings");
    }

    #[test]
    fn postings_spill_round_trip() {
        let dir = SpillDir::new().unwrap();
        let p = naming::postings_path(dir.path(), 0, false);

        let recs = vec![
            PostingRecord {
                tri_hash: 0x1111_1111,
                doc_id: 0,
                line_no: 1,
                byte_offset: 0,
            },
            PostingRecord {
                tri_hash: 0x1111_1111,
                doc_id: 0,
                line_no: 2,
                byte_offset: 10,
            },
            PostingRecord {
                tri_hash: 0x2222_2222,
                doc_id: 1,
                line_no: 1,
                byte_offset: 0,
            },
            PostingRecord {
                tri_hash: 0x3333_3333,
                doc_id: 7,
                line_no: 99,
                byte_offset: 1234,
            },
        ];
        let mut w = PostingsSpillWriter::create(&p, TEST_BUF).unwrap();
        w.push_slice(&recs).unwrap();
        let n = w.finish().unwrap();
        assert_eq!(n, recs.len() as u64);

        let r = PostingsSpillReader::open(&p, TEST_BUF).unwrap();
        let got: Vec<PostingRecord> = r.map(|x| x.unwrap()).collect();
        assert_eq!(got.len(), recs.len());
        // Round-trip is in insertion order — sort both lists and compare
        // because reader doesn't re-sort (the k-way merge would).
        let mut got_sorted = got;
        got_sorted.sort_by_key(|r| (r.tri_hash, r.doc_id, r.line_no, r.byte_offset));
        let mut recs_sorted = recs;
        recs_sorted.sort_by_key(|r| (r.tri_hash, r.doc_id, r.line_no, r.byte_offset));
        assert_eq!(got_sorted, recs_sorted);
    }

    #[test]
    fn postings_spill_rejects_bad_magic() {
        let dir = SpillDir::new().unwrap();
        let p = dir.path().join("bad.postings.spill");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(b"NOTASPILL__________").unwrap();
        }
        let err = PostingsSpillReader::open(&p, TEST_BUF).unwrap_err();
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn masks_spill_round_trip() {
        let dir = SpillDir::new().unwrap();
        let p = naming::masks_path(dir.path(), 42, false);

        let entries = vec![
            MaskSpillEntry {
                tri_hash: 1,
                mask: [0xFFFF_FFFF; 8],
            },
            MaskSpillEntry {
                tri_hash: 2,
                mask: [0; 8],
            },
            MaskSpillEntry {
                tri_hash: 100,
                mask: {
                    let mut m = [0u32; 8];
                    m[0] = 1;
                    m[7] = 0x8000_0000;
                    m
                },
            },
        ];
        let mut w = MasksSpillWriter::create(&p, TEST_BUF).unwrap();
        w.push_slice(&entries).unwrap();
        w.finish().unwrap();

        let r = MasksSpillReader::open(&p, TEST_BUF).unwrap();
        let got: Vec<MaskSpillEntry> = r.map(|x| x.unwrap()).collect();
        assert_eq!(got.len(), entries.len());
        for (a, b) in got.iter().zip(entries.iter()) {
            assert_eq!(a.tri_hash, b.tri_hash);
            assert_eq!(a.mask, b.mask);
        }
    }

    #[test]
    fn masks_spill_rejects_bad_magic() {
        let dir = SpillDir::new().unwrap();
        let p = dir.path().join("bad.masks.spill");
        {
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(&[0u8; 16]).unwrap();
        }
        let err = MasksSpillReader::open(&p, TEST_BUF).unwrap_err();
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn spill_dir_is_removed_on_drop() {
        let path_holder;
        {
            let d = SpillDir::new().unwrap();
            path_holder = d.path().to_path_buf();
            assert!(path_holder.exists());
        }
        assert!(!path_holder.exists(), "SpillDir Drop should remove its dir");
    }

    #[test]
    fn chunk_outputs_can_be_cleared() {
        let mut out = ChunkOutputs::new();
        extract_file(b"hello\nworld", 0, true, &mut out);
        assert!(!out.postings.is_empty());
        assert!(!out.postings_ci.is_empty());
        out.clear();
        assert!(out.postings.is_empty());
        assert!(out.postings_ci.is_empty());
        assert!(out.masks.is_empty());
        assert!(out.masks_ci.is_empty());
    }

    /// Build two synthetic postings spills + two synthetic masks spills, run
    /// `merge_to_store`, then verify the on-disk output round-trips cleanly.
    /// This is the load-bearing test for M2 — if the k-way merge is wrong,
    /// the lookup table is unsorted or the postings are not deduped, this
    /// test catches it.
    ///
    /// Test corpus: trigram X has postings in both spills (cross-spill
    /// dedup); trigram Y appears only in spill 0; trigram Z appears only
    /// in spill 1. Each spill is sorted ascending by `(tri_hash, doc_id,
    /// line_no, byte_offset)` — the merge depends on this invariant.
    #[test]
    fn merge_round_trip_two_spills() {
        let spill = SpillDir::new().unwrap();
        let output = tempfile::TempDir::new().unwrap();
        let cfg = StreamingConfig::defaults();

        let x_hash = crc32fast::hash(b"the");
        let y_hash = crc32fast::hash(b"abc");
        let z_hash = crc32fast::hash(b"foo");
        let hashes = [x_hash, y_hash, z_hash];
        let mut hashes_sorted = hashes;
        hashes_sorted.sort_unstable();
        let y_is_smallest = hashes_sorted[0] == y_hash;

        // Spill 0, sorted ascending by (tri_hash, doc, line, off). Y first
        // (smaller hash), then X entries, ascending by (doc, line).
        let p1_path = naming::postings_path(spill.path(), 0, false);
        let mut w = PostingsSpillWriter::create(&p1_path, TEST_BUF).unwrap();
        if y_is_smallest {
            w.push(PostingRecord {
                tri_hash: y_hash,
                doc_id: 1,
                line_no: 1,
                byte_offset: 0,
            })
            .unwrap();
            w.push(PostingRecord {
                tri_hash: x_hash,
                doc_id: 5,
                line_no: 7,
                byte_offset: 100,
            })
            .unwrap();
            w.push(PostingRecord {
                tri_hash: x_hash,
                doc_id: 5,
                line_no: 9,
                byte_offset: 200,
            })
            .unwrap();
        } else {
            w.push(PostingRecord {
                tri_hash: x_hash,
                doc_id: 5,
                line_no: 7,
                byte_offset: 100,
            })
            .unwrap();
            w.push(PostingRecord {
                tri_hash: x_hash,
                doc_id: 5,
                line_no: 9,
                byte_offset: 200,
            })
            .unwrap();
            w.push(PostingRecord {
                tri_hash: y_hash,
                doc_id: 1,
                line_no: 1,
                byte_offset: 0,
            })
            .unwrap();
        }
        w.finish().unwrap();

        // Spill 1: X first (smaller than Z), then Z.
        let p2_path = naming::postings_path(spill.path(), 1, false);
        let mut w = PostingsSpillWriter::create(&p2_path, TEST_BUF).unwrap();
        if x_hash < z_hash {
            w.push(PostingRecord {
                tri_hash: x_hash,
                doc_id: 10,
                line_no: 3,
                byte_offset: 50,
            })
            .unwrap();
            w.push(PostingRecord {
                tri_hash: z_hash,
                doc_id: 99,
                line_no: 1,
                byte_offset: 0,
            })
            .unwrap();
        } else {
            w.push(PostingRecord {
                tri_hash: z_hash,
                doc_id: 99,
                line_no: 1,
                byte_offset: 0,
            })
            .unwrap();
            w.push(PostingRecord {
                tri_hash: x_hash,
                doc_id: 10,
                line_no: 3,
                byte_offset: 50,
            })
            .unwrap();
        }
        w.finish().unwrap();

        // Masks: X gets two contributors; Y/Z each get one. Each mask spill
        // is sorted ascending by tri_hash.
        let m1_path = naming::masks_path(spill.path(), 0, false);
        let m2_path = naming::masks_path(spill.path(), 1, false);

        let mut x_part_a = [0u32; 8];
        x_part_a[0] = 0b101;
        let x_part_b = [0u32; 8];
        let mut y_mask = [0u32; 8];
        y_mask[3] = 0xFFFF_FFFF;
        let mut z_mask = [0u32; 8];
        z_mask[7] = 0x8000_0000;

        let mut w = MasksSpillWriter::create(&m1_path, TEST_BUF).unwrap();
        let mut spill0_pairs = vec![(x_hash, x_part_a), (y_hash, y_mask)];
        spill0_pairs.sort_by_key(|(h, _)| *h);
        for (h, m) in spill0_pairs {
            w.push(&MaskSpillEntry {
                tri_hash: h,
                mask: m,
            })
            .unwrap();
        }
        w.finish().unwrap();

        let mut w = MasksSpillWriter::create(&m2_path, TEST_BUF).unwrap();
        let mut spill1_pairs = vec![(x_hash, x_part_b), (z_hash, z_mask)];
        spill1_pairs.sort_by_key(|(h, _)| *h);
        for (h, m) in spill1_pairs {
            w.push(&MaskSpillEntry {
                tri_hash: h,
                mask: m,
            })
            .unwrap();
        }
        w.finish().unwrap();

        let result = merge_to_store(
            output.path(),
            "ngrams",
            &[p1_path, p2_path],
            &[m1_path, m2_path],
            &cfg,
        )
        .unwrap();

        assert_eq!(result.num_trigrams, 3, "three unique trigrams expected");
        assert!(result.postings_len > 0);

        let lookup_bytes = std::fs::read(output.path().join("ngrams.lookup")).unwrap();
        let entries: Vec<u32> = {
            let mut out = Vec::new();
            for chunk in lookup_bytes.chunks(16) {
                let h = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(h);
            }
            out
        };
        for w in entries.windows(2) {
            assert!(w[0] <= w[1], "postings lookup not sorted: {:?}", entries);
        }
        assert_eq!(entries.len(), 3);

        let mask_lookup_bytes = std::fs::read(output.path().join("ngrams.masks.lookup")).unwrap();
        let mask_entries: Vec<u32> = {
            let mut out = Vec::new();
            for chunk in mask_lookup_bytes.chunks(16) {
                let h = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(h);
            }
            out
        };
        assert_eq!(
            mask_entries, entries,
            "mask lookup must follow postings order"
        );

        let postings_data = std::fs::read(output.path().join("ngrams.postings")).unwrap();

        let mut x_off = 0usize;
        let mut x_len = 0u32;
        for chunk in lookup_bytes.chunks(16) {
            let h = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let off = u64::from_le_bytes([
                chunk[4], chunk[5], chunk[6], chunk[7], chunk[8], chunk[9], chunk[10], chunk[11],
            ]) as usize;
            let len = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
            if h == x_hash {
                x_off = off;
                x_len = len;
                break;
            }
        }
        let x_postings: Vec<(u32, u32, u32)> =
            crate::postenc::PostingReader::new(&postings_data[x_off..x_off + x_len as usize])
                .collect();
        let mut x_keys: Vec<(u32, u32, u32)> =
            x_postings.iter().map(|(d, l, o)| (*d, *l, *o)).collect();
        x_keys.sort_unstable();
        let mut want = vec![(5u32, 7u32, 100u32), (5, 9, 200), (10, 3, 50)];
        want.sort_unstable();
        assert_eq!(x_keys, want, "X postings mismatch");

        let bm_lookup_bytes = std::fs::read(output.path().join("ngrams.bitmaps.lookup")).unwrap();
        let bitmaps_data = std::fs::read(output.path().join("ngrams.bitmaps")).unwrap();
        let mut x_bm_off = 0usize;
        let mut x_bm_len = 0u32;
        for chunk in bm_lookup_bytes.chunks(16) {
            let h = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let off = u64::from_le_bytes([
                chunk[4], chunk[5], chunk[6], chunk[7], chunk[8], chunk[9], chunk[10], chunk[11],
            ]) as usize;
            let len = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
            if h == x_hash {
                x_bm_off = off;
                x_bm_len = len;
                break;
            }
        }
        let x_bitmap = roaring::RoaringBitmap::deserialize_from(
            &bitmaps_data[x_bm_off..x_bm_off + x_bm_len as usize],
        )
        .unwrap();
        let mut x_docs: Vec<u32> = x_bitmap.iter().collect();
        x_docs.sort_unstable();
        assert_eq!(x_docs, vec![5, 10], "X bitmap mismatch");

        let mask_offset: usize = {
            let mut off = 0usize;
            for chunk in mask_lookup_bytes.chunks(16) {
                let h = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let o = u64::from_le_bytes([
                    chunk[4], chunk[5], chunk[6], chunk[7], chunk[8], chunk[9], chunk[10],
                    chunk[11],
                ]) as usize;
                if h == x_hash {
                    off = o;
                    break;
                }
            }
            off
        };
        let masks_data = std::fs::read(output.path().join("ngrams.masks")).unwrap();
        let x_mask_bytes = &masks_data[mask_offset..mask_offset + 32];
        let x_mask_w0 = u32::from_le_bytes([
            x_mask_bytes[0],
            x_mask_bytes[1],
            x_mask_bytes[2],
            x_mask_bytes[3],
        ]);
        assert_eq!(
            x_mask_w0, 0b101,
            "X mask word 0 should be OR of both spills (a|c)"
        );
    }

    #[test]
    fn merge_handles_empty_spills() {
        let spill = SpillDir::new().unwrap();
        let output = tempfile::TempDir::new().unwrap();
        let cfg = StreamingConfig::defaults();

        // Two empty postings spills + two empty masks spills.
        let p1 = naming::postings_path(spill.path(), 0, false);
        let p2 = naming::postings_path(spill.path(), 1, false);
        let m1 = naming::masks_path(spill.path(), 0, false);
        let m2 = naming::masks_path(spill.path(), 1, false);
        PostingsSpillWriter::create(&p1, TEST_BUF)
            .unwrap()
            .finish()
            .unwrap();
        PostingsSpillWriter::create(&p2, TEST_BUF)
            .unwrap()
            .finish()
            .unwrap();
        MasksSpillWriter::create(&m1, TEST_BUF)
            .unwrap()
            .finish()
            .unwrap();
        MasksSpillWriter::create(&m2, TEST_BUF)
            .unwrap()
            .finish()
            .unwrap();

        let result = merge_to_store(output.path(), "ngrams", &[p1, p2], &[m1, m2], &cfg).unwrap();

        assert_eq!(result.num_trigrams, 0);
        assert_eq!(result.postings_len, 0);

        // Files exist and are empty (no lookup entries).
        let lookup = std::fs::metadata(output.path().join("ngrams.lookup")).unwrap();
        assert_eq!(lookup.len(), 0);
    }

    /// Round-trip a small corpus via the full `streaming_build_from_paths`
    /// pipeline: chunk → extract → spill → merge. Verify the produced
    /// `ngrams.*` files are well-formed (lookup sorted, postings + masks
    /// consistent with what the legacy `SparseIndex` would produce for the
    /// same content). This is the M3 glue-level smoke test.
    #[test]
    fn streaming_build_round_trip_small_corpus() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p1 = tmp.path().join("a.txt");
        let p2 = tmp.path().join("b.txt");
        let p3 = tmp.path().join("c.txt");
        std::fs::write(&p1, b"hello world\nfoo bar\n").unwrap();
        std::fs::write(&p2, b"hello again\nbaz qux\n").unwrap();
        // Duplicate a large chunk of p2 inside p3 to exercise cross-doc dedup.
        std::fs::write(&p3, b"hello again\nfoo bar\n").unwrap();
        let paths = vec![p1.clone(), p2.clone(), p3.clone()];

        let output = tempfile::TempDir::new().unwrap();
        let cfg = StreamingConfig::defaults();
        let result = streaming_build_from_paths(&paths, false, &cfg, output.path()).unwrap();

        assert_eq!(result.num_docs, 3);
        assert!(result.num_ngrams > 0);
        assert!(result.postings_len > 0);

        // All six CS index files exist.
        for suffix in [
            "ngrams.postings",
            "ngrams.lookup",
            "ngrams.bitmaps",
            "ngrams.bitmaps.lookup",
            "ngrams.masks",
            "ngrams.masks.lookup",
        ] {
            assert!(
                output.path().join(suffix).exists(),
                "missing expected output file: {}",
                suffix
            );
        }

        // Lookups are sorted by tri_hash and the three (postings, bitmap, mask)
        // lookup files use identical hash orders (binary-search invariant).
        let post_lk = std::fs::read(output.path().join("ngrams.lookup")).unwrap();
        let bm_lk = std::fs::read(output.path().join("ngrams.bitmaps.lookup")).unwrap();
        let mask_lk = std::fs::read(output.path().join("ngrams.masks.lookup")).unwrap();
        assert_eq!(
            post_lk.len(),
            bm_lk.len(),
            "postings/bitmap lookup must align"
        );
        assert_eq!(
            post_lk.len(),
            mask_lk.len(),
            "postings/mask lookup must align"
        );
        for (a, b) in post_lk.chunks(16).zip(bm_lk.chunks(16)) {
            assert_eq!(
                &a[..4],
                &b[..4],
                "hash key must match between postings and bitmap lookups",
            );
        }
        for w in post_lk.chunks(16).collect::<Vec<_>>().windows(2) {
            let h0 = u32::from_le_bytes([w[0][0], w[0][1], w[0][2], w[0][3]]);
            let h1 = u32::from_le_bytes([w[1][0], w[1][1], w[1][2], w[1][3]]);
            assert!(h0 <= h1, "postings lookup not sorted");
        }

        // The trigram "hel" appears in all three files. Its doc set should be
        // exactly {0, 1, 2}.
        let hel_hash = crc32fast::hash(b"hel");
        let hel_bm_off: u64 = bm_lk
            .chunks(16)
            .map(|c| {
                let h = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let off = u64::from_le_bytes([c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]]);
                (h, off)
            })
            .find(|(h, _)| *h == hel_hash)
            .map(|(_, off)| off)
            .expect("hel trigram missing from bitmap lookup");

        let bm_data = std::fs::read(output.path().join("ngrams.bitmaps")).unwrap();
        // Bitmap len lives at offset 12 of the lookup row.
        let bm_len: u32 = bm_lk
            .chunks(16)
            .find(|c| {
                let h = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                h == hel_hash
            })
            .map(|c| u32::from_le_bytes([c[12], c[13], c[14], c[15]]))
            .unwrap();
        let hel_bm = roaring::RoaringBitmap::deserialize_from(
            &bm_data[hel_bm_off as usize..(hel_bm_off as usize + bm_len as usize)],
        )
        .unwrap();
        let mut docs: Vec<u32> = hel_bm.iter().collect();
        docs.sort_unstable();
        assert_eq!(docs, vec![0, 1, 2], "hel doc set");
    }

    /// Case-insensitive streaming build produces six CS files PLUS six CI
    /// files with matching filenames. This is the simplest CI-companion
    /// smoke check — the M3 plumbing must call `merge_to_store` twice.
    #[test]
    fn streaming_build_produces_ci_companion() {
        let tmp = tempfile::TempDir::new().unwrap();
        let p1 = tmp.path().join("a.txt");
        std::fs::write(&p1, b"Hello WORLD\nFoo bar\n").unwrap();
        let paths = vec![p1];

        let output = tempfile::TempDir::new().unwrap();
        let cfg = StreamingConfig::defaults();
        let _ = streaming_build_from_paths(&paths, true, &cfg, output.path()).unwrap();

        for suffix in [
            "ngrams.postings",
            "ngrams.lookup",
            "ngrams.bitmaps",
            "ngrams.bitmaps.lookup",
            "ngrams.masks",
            "ngrams.masks.lookup",
            "ngrams.ci.postings",
            "ngrams.ci.lookup",
            "ngrams.ci.bitmaps",
            "ngrams.ci.bitmaps.lookup",
            "ngrams.ci.masks",
            "ngrams.ci.masks.lookup",
        ] {
            assert!(
                output.path().join(suffix).exists(),
                "missing CI companion file: {}",
                suffix
            );
        }

        // The CI map must hold the folded "hel", not the original-case "Hel".
        let hel_hash = crc32fast::hash(b"hel");
        let helm_h_hash = crc32fast::hash(b"Hel");
        let ci_post_lk = std::fs::read(output.path().join("ngrams.ci.lookup")).unwrap();
        let hashes: Vec<u32> = ci_post_lk
            .chunks(16)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            hashes.contains(&hel_hash),
            "folded hel must appear in CI companion"
        );
        assert!(
            !hashes.contains(&helm_h_hash),
            "un-folded Hel must NOT appear in CI companion"
        );
    }
}
