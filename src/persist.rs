use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ignore::WalkBuilder;
use memmap2::Mmap;
use rayon::prelude::*;

use roaring::RoaringBitmap;

use crate::casefold;
use crate::index::{FollowerMask, Posting, SparseIndex, TrigramBuilder};
use crate::postenc::{PostingReader, PostingWriter};
use crate::trigram;

/// On-disk mask entry size (8 × u32 little-endian = 256 bits).
const MASK_ENTRY_SIZE: usize = 32;

/// On-disk index format version. Bumped to 4 for the compact (delta-varint)
/// posting format; the loader transparently migrates older indices by
/// rebuilding from the recorded `root_dir`.
pub const INDEX_VERSION: u32 = 4;

// --- Zero-copy read helpers ---

#[inline(always)]
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline(always)]
fn read_u64_le(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
        data[off + 4],
        data[off + 5],
        data[off + 6],
        data[off + 7],
    ])
}

/// Sorted merge intersection of two sorted line-posting slices.
/// Intersects on (doc_id, line_no), keeps byte_offset from `a`.
fn sorted_intersect_lines(a: &[(u32, u32, u32)], b: &[(u32, u32, u32)]) -> Vec<(u32, u32, u32)> {
    let mut result = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let ka = (a[i].0, a[i].1);
        let kb = (b[j].0, b[j].1);
        match ka.cmp(&kb) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result
}

/// Merge two sorted line-posting slices into a new sorted Vec (union).
fn merge_sorted_lines(a: &[(u32, u32, u32)], b: &[(u32, u32, u32)]) -> Vec<(u32, u32, u32)> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let ka = (a[i].0, a[i].1);
        let kb = (b[j].0, b[j].1);
        match ka.cmp(&kb) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// --- Timing ---

pub struct SearchTiming {
    pub lookup_ms: f64,
    pub bitmap_intersect_ms: f64,
    pub verify_ms: f64,
    pub candidates: usize,
    pub matches: usize,
    /// Verify strategy used: "line-level", "file-level", or "file-level (fallback)"
    pub strategy: String,
    /// Match density (lines per file) that drove the strategy decision
    pub density: f64,
    /// Number of candidates eliminated by the 4-byte line prefix filter (no I/O)
    pub prefix_filtered: usize,
}

// --- Line-level search result ---

pub struct LineHit<'a> {
    pub path: &'a Path,
    pub line_no: u32,
    pub byte_offset: u32,
}

/// Search result from the index: either precise line-level hits or fallback to all files.
pub enum SearchResult<'a> {
    /// Line-level candidates from trigram intersection
    LineHits(Vec<LineHit<'a>>),
    /// Bitmap-only candidates: selective bitmap AND produced few files, skip posting load
    BitmapFiles(Vec<&'a Path>),
    /// Fallback: all files (pattern too short for trigrams)
    AllFiles(Vec<&'a Path>),
}

// --- Index structures ---

#[derive(serde::Serialize, serde::Deserialize)]
pub struct IndexMeta {
    pub version: u32,
    pub num_docs: usize,
    pub num_ngrams: usize,
    pub root_dir: String,
    pub built_at: String,
    pub file_mtimes: HashMap<String, u64>,
    /// Directory mtimes — used for fast stale detection without walking the tree.
    /// A changed dir mtime means files were added, deleted, or renamed in that dir.
    #[serde(default)]
    pub dir_mtimes: HashMap<String, u64>,
    /// Number of docs in the main (non-delta) index. Set on full build.
    #[serde(default)]
    pub main_num_docs: Option<usize>,
    /// Whether this index was built over case-folded text (a case-insensitive
    /// index). `false` (default) = case-sensitive. Set when building the CI
    /// index so `-i` searches can route to it. Reserved in v4 for the planned
    /// dual case-sensitive / case-insensitive index pair.
    #[serde(default)]
    pub case_insensitive: bool,
}

#[derive(Clone)]
pub struct LookupEntry {
    pub hash: u32,
    pub offset: u64,
    pub len: u32,
}

const LOOKUP_ENTRY_SIZE: usize = 4 + 8 + 4; // 16 bytes

pub struct PersistentIndex {
    /// Main lookup table — memory-mapped for zero-copy binary search
    pub lookup_mmap: Mmap,
    pub lookup_count: usize,
    /// Main postings — memory-mapped
    pub postings_mmap: Mmap,
    /// Roaring bitmap file — memory-mapped (one serialized bitmap per trigram)
    pub bitmap_mmap: Option<Mmap>,
    /// Bitmap lookup table — memory-mapped (hash → offset/len into bitmap_mmap)
    pub bitmap_lookup_mmap: Option<Mmap>,
    pub bitmap_lookup_count: usize,
    /// Doc IDs: mmap'd flat buffer + offset table for zero-alloc load
    pub docids_mmap: Mmap,
    pub docid_offsets: Vec<(u32, u16)>, // (offset, length) for main docs
    pub delta_doc_ids: Vec<PathBuf>,    // delta docs (small count)
    pub meta: IndexMeta,
    // Overlay for incremental updates
    pub deleted_docs: HashSet<u32>,
    pub delta_lookup: Vec<LookupEntry>,
    pub delta_postings: Vec<u8>,
    pub main_num_docs: usize,
    // Case-insensitive companion index (`ngrams.ci.*`), present only when the
    // index was built with `-i`. Same docids/delta-docids/deleted set as the
    // case-sensitive index; only the trigram postings/bitmaps differ (folded).
    pub lookup_ci_mmap: Option<Mmap>,
    pub lookup_ci_count: usize,
    pub postings_ci_mmap: Option<Mmap>,
    pub bitmap_ci_mmap: Option<Mmap>,
    pub bitmap_lookup_ci_mmap: Option<Mmap>,
    pub bitmap_lookup_ci_count: usize,
    pub delta_lookup_ci: Vec<LookupEntry>,
    pub delta_postings_ci: Vec<u8>,
    // Successor masks (Cursor phrase-aware trigram pre-filter). Optional —
    // older indexes and indexes that opted out don't have them. When
    // `masks_mmap` is `None`, `mask_overlap` returns true unconditionally
    // (no pruning) so the searcher can be written as if masks were always
    // present.
    pub masks_mmap: Option<Mmap>,
    pub masks_lookup_mmap: Option<Mmap>,
    #[allow(dead_code)] // read by `lookup_mask`; the count lives in the struct for symmetry
    pub masks_count: usize,
    pub masks_ci_mmap: Option<Mmap>,
    pub masks_ci_lookup_mmap: Option<Mmap>,
    #[allow(dead_code)]
    pub masks_ci_count: usize,
}

impl PersistentIndex {
    /// Release the memory-mapped files backing this index. After calling this,
    /// none of the search methods are usable. Use this when you need to
    /// overwrite or rename the index files on Windows, where the OS refuses
    /// to truncate or rename a file that still has an active mapping. The
    /// safe pattern is to call this immediately before `compact()` or any
    /// other operation that writes to the index directory.
    pub fn close(&mut self) {
        // Replace each file-backed mapping with a mapping over a throwaway
        // scratch file. Assigning to `self.lookup_mmap` (etc.) drops the prior
        // `Mmap` first, which is what releases the OS section on Windows.
        // The scratch files live in the system temp dir and are deleted
        // immediately; we keep their file handles alive for the lifetime of
        // the mmap by leaking them. (The OS reaps them when the process
        // exits, by which point nothing in this index is in use anyway.)
        fn empty_mmap() -> Mmap {
            use std::sync::atomic::{AtomicU64, Ordering};
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "fgr-empty-mmap-{}-{}.bin",
                std::process::id(),
                n
            ));
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .expect("create scratch mmap file");
            f.set_len(1).expect("set scratch length");
            let _ = std::fs::remove_file(&path);
            let mmap = unsafe { Mmap::map(&f).expect("mmap scratch file") };
            std::mem::forget(f);
            mmap
        }

        self.lookup_mmap = empty_mmap();
        self.postings_mmap = empty_mmap();
        self.docids_mmap = empty_mmap();
        if self.bitmap_mmap.is_some() {
            self.bitmap_mmap = Some(empty_mmap());
        }
        if self.bitmap_lookup_mmap.is_some() {
            self.bitmap_lookup_mmap = Some(empty_mmap());
        }
        if self.masks_mmap.is_some() {
            self.masks_mmap = Some(empty_mmap());
        }
        if self.masks_lookup_mmap.is_some() {
            self.masks_lookup_mmap = Some(empty_mmap());
        }
        if self.lookup_ci_mmap.is_some() {
            self.lookup_ci_mmap = Some(empty_mmap());
        }
        if self.postings_ci_mmap.is_some() {
            self.postings_ci_mmap = Some(empty_mmap());
        }
        if self.bitmap_ci_mmap.is_some() {
            self.bitmap_ci_mmap = Some(empty_mmap());
        }
        if self.bitmap_lookup_ci_mmap.is_some() {
            self.bitmap_lookup_ci_mmap = Some(empty_mmap());
        }
        if self.masks_ci_mmap.is_some() {
            self.masks_ci_mmap = Some(empty_mmap());
        }
        if self.masks_ci_lookup_mmap.is_some() {
            self.masks_ci_lookup_mmap = Some(empty_mmap());
        }
    }
    pub fn doc_path(&self, id: u32) -> Option<&Path> {
        let id = id as usize;
        if id < self.docid_offsets.len() {
            let (off, len) = self.docid_offsets[id];
            let end = off as usize + len as usize;
            if end <= self.docids_mmap.len() {
                let bytes = &self.docids_mmap[off as usize..end];
                std::str::from_utf8(bytes).ok().map(Path::new)
            } else {
                None
            }
        } else {
            let delta_idx = id - self.docid_offsets.len();
            self.delta_doc_ids.get(delta_idx).map(|p| p.as_path())
        }
    }

    /// Total number of docs (main + delta).
    pub fn num_docs(&self) -> usize {
        self.docid_offsets.len() + self.delta_doc_ids.len()
    }

    /// Whether a case-insensitive companion index is loaded.
    #[inline]
    pub fn has_ci(&self) -> bool {
        self.postings_ci_mmap.is_some()
    }

    // --- Low-level lookup methods (zero-copy) ---
    //
    // Each takes `ci: bool` to select the case-sensitive store (the default
    // `ngrams.*` mmaps) or the case-insensitive companion (`ngrams.ci.*`). The
    // search code threads the same flag through so an `(?i)` query resolves
    // entirely against the folded store.

    /// Binary search in the mmap'd lookup table (CS or CI).
    #[inline]
    fn find_in_main_lookup(&self, hash: u32, ci: bool) -> Option<(u64, u32)> {
        let (data, count) = if ci {
            (&**self.lookup_ci_mmap.as_ref()?, self.lookup_ci_count)
        } else {
            (&*self.lookup_mmap, self.lookup_count)
        };
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let h = read_u32_le(data, mid * LOOKUP_ENTRY_SIZE);
            match h.cmp(&hash) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let off = read_u64_le(data, mid * LOOKUP_ENTRY_SIZE + 4);
                    let len = read_u32_le(data, mid * LOOKUP_ENTRY_SIZE + 12);
                    return Some((off, len));
                }
            }
        }
        None
    }

    /// Get raw posting bytes for a trigram hash from the main index (CS or CI).
    #[inline]
    fn main_posting_data(&self, hash: u32, ci: bool) -> Option<&[u8]> {
        let (offset, len) = self.find_in_main_lookup(hash, ci)?;
        let mmap = if ci {
            &**self.postings_ci_mmap.as_ref()?
        } else {
            &*self.postings_mmap
        };
        let start = offset as usize;
        let end = start + len as usize;
        if end <= mmap.len() {
            Some(&mmap[start..end])
        } else {
            None
        }
    }

    /// Get raw posting bytes for a trigram hash from the delta index (CS or CI).
    #[inline]
    fn delta_posting_data(&self, hash: u32, ci: bool) -> Option<&[u8]> {
        let (lookup, postings) = if ci {
            (&self.delta_lookup_ci, &self.delta_postings_ci)
        } else {
            (&self.delta_lookup, &self.delta_postings)
        };
        if lookup.is_empty() {
            return None;
        }
        let idx = lookup.binary_search_by_key(&hash, |e| e.hash).ok()?;
        let entry = &lookup[idx];
        let start = entry.offset as usize;
        let end = start + entry.len as usize;
        if end <= postings.len() {
            Some(&postings[start..end])
        } else {
            None
        }
    }

    // --- Successor-mask lookup (Tier 0.5: 4-byte pre-filter) ---
    //
    // The mask is a 256-bit (8 × u32 LE) bitmap per trigram stored
    // verbatim in the order of the trigram list. The lookup table is the
    // same shape as the postings lookup (hash → offset/len), so we
    // reuse the binary-search helper. We never deserialize — a mask is
    // just 8 u32 reads, so we return the borrowed `&[u32; 8]` slice and
    // let the caller index into it.

    /// Return the 8-word (256-bit) successor mask for `hash`, or `None` if
    /// the trigram isn't present. When the index has no mask files (older
    /// build, or migrated index), this also returns `None`.
    #[inline]
    #[allow(dead_code)] // exercised by the mask test target; not yet wired into search_timed
    fn lookup_mask(&self, hash: u32, ci: bool) -> Option<FollowerMask> {
        let (data, lookup, count) = if ci {
            (
                self.masks_ci_mmap.as_deref()?,
                self.masks_ci_lookup_mmap.as_deref()?,
                self.masks_ci_count,
            )
        } else {
            (
                self.masks_mmap.as_deref()?,
                self.masks_lookup_mmap.as_deref()?,
                self.masks_count,
            )
        };
        let (offset, _len) = self.find_in_masks_lookup(hash, lookup, count)?;
        let off = offset as usize;
        // Each mask is exactly MASK_ENTRY_SIZE bytes (8 × u32 LE). Copy
        // out into a stack-allocated FollowerMask — the cost (32 bytes
        // per call) is in the noise next to the binary search that
        // preceded it, and avoids needing an alignment proof for an
        // unsafe slice cast through a fat pointer.
        let bytes = &data[off..off + MASK_ENTRY_SIZE];
        let mut out = [0u32; 8];
        for (i, w) in out.iter_mut().enumerate() {
            let b = i * 4;
            *w = u32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
        }
        Some(out)
    }

    /// Binary search in a mask lookup table. The layout mirrors
    /// `find_in_main_lookup` (16-byte entries: hash, offset, len) but we
    /// take the data + count explicitly so the same code services both
    /// the CS and CI mask files.
    #[inline]
    #[allow(dead_code)]
    fn find_in_masks_lookup(&self, hash: u32, data: &[u8], count: usize) -> Option<(u64, u32)> {
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let h = read_u32_le(data, mid * LOOKUP_ENTRY_SIZE);
            match h.cmp(&hash) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let off = read_u64_le(data, mid * LOOKUP_ENTRY_SIZE + 4);
                    let len = read_u32_le(data, mid * LOOKUP_ENTRY_SIZE + 12);
                    return Some((off, len));
                }
            }
        }
        None
    }

    /// Returns true when at least one byte in `allowed_bytes` is recorded
    /// as a successor of the trigram identified by `hash` somewhere in
    /// the corpus. This is the "4-byte query" half of the phrase-aware
    /// trigram pre-filter: given a regex literal run, the searcher asks
    /// "could this trigram plausibly be followed by any of the bytes the
    /// regex requires next?" — if no, the alternative is provably empty
    /// and can be skipped without touching the postings.
    ///
    /// The CI variant resolves against the case-folded companion mask;
    /// pass `ci = true` when the query is `(?i)`.
    ///
    /// When the loaded index has no mask files (older version, or the
    /// optional mask wasn't built), returns true unconditionally so the
    /// searcher can use the API as if masks were always present.
    #[inline]
    #[allow(dead_code)] // exposed for the searcher to call; not yet wired in
    pub fn mask_overlap(&self, hash: u32, allowed_bytes: &[u8], ci: bool) -> bool {
        // No mask files → no pruning (preserves backward compatibility).
        let mask = match self.lookup_mask(hash, ci) {
            Some(m) => m,
            None => return true,
        };
        for &b in allowed_bytes {
            let word = (b >> 5) as usize;
            let bit = b & 0x1F;
            if mask[word] & (1u32 << bit) != 0 {
                return true;
            }
        }
        false
    }

    // --- Roaring bitmap lookup (Tier 1) ---

    /// Binary search in the mmap'd bitmap lookup table (CS or CI).
    #[inline]
    fn find_in_bitmap_lookup(&self, hash: u32, ci: bool) -> Option<(u64, u32)> {
        let (data, count) = if ci {
            (
                self.bitmap_lookup_ci_mmap.as_ref()?,
                self.bitmap_lookup_ci_count,
            )
        } else {
            (self.bitmap_lookup_mmap.as_ref()?, self.bitmap_lookup_count)
        };
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let h = read_u32_le(data, mid * LOOKUP_ENTRY_SIZE);
            match h.cmp(&hash) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let off = read_u64_le(data, mid * LOOKUP_ENTRY_SIZE + 4);
                    let len = read_u32_le(data, mid * LOOKUP_ENTRY_SIZE + 12);
                    return Some((off, len));
                }
            }
        }
        None
    }

    /// Deserialize a RoaringBitmap for a trigram hash from the bitmap mmap.
    /// Uses unchecked deserialization for speed (data was written by us).
    #[inline]
    fn lookup_bitmap(&self, hash: u32, ci: bool) -> Option<RoaringBitmap> {
        let (offset, len) = self.find_in_bitmap_lookup(hash, ci)?;
        let bm_data = if ci {
            self.bitmap_ci_mmap.as_ref()?
        } else {
            self.bitmap_mmap.as_ref()?
        };
        let start = offset as usize;
        let end = start + len as usize;
        if end > bm_data.len() {
            return None;
        }
        RoaringBitmap::deserialize_unchecked_from(&bm_data[start..end]).ok()
    }

    /// Extract sorted line postings from a compact posting blob, excluding deleted docs.
    fn extract_line_postings(&self, data: &[u8]) -> Vec<(u32, u32, u32)> {
        let mut postings = Vec::new();
        for (doc_id, line_no, byte_offset) in PostingReader::new(data) {
            if !self.deleted_docs.contains(&doc_id) {
                postings.push((doc_id, line_no, byte_offset));
            }
        }
        postings
    }

    /// Extract sorted line postings, filtering to only doc_ids in the bitmap.
    fn extract_line_postings_filtered(
        &self,
        data: &[u8],
        filter: &RoaringBitmap,
    ) -> Vec<(u32, u32, u32)> {
        let mut postings = Vec::new();
        for (doc_id, line_no, byte_offset) in PostingReader::new(data) {
            if filter.contains(doc_id) && !self.deleted_docs.contains(&doc_id) {
                postings.push((doc_id, line_no, byte_offset));
            }
        }
        postings
    }

    /// Get merged (main + delta) sorted line postings for a trigram hash,
    /// filtered to only doc_ids in the candidate bitmap.
    fn trigram_line_postings_filtered(
        &self,
        hash: u32,
        filter: &RoaringBitmap,
        ci: bool,
    ) -> Option<Vec<(u32, u32, u32)>> {
        let main = self.main_posting_data(hash, ci);
        let delta = self.delta_posting_data(hash, ci);

        if main.is_none() && delta.is_none() {
            return None;
        }

        let main_postings = main
            .map(|d| self.extract_line_postings_filtered(d, filter))
            .unwrap_or_default();
        let delta_postings = delta
            .map(|d| self.extract_line_postings_filtered(d, filter))
            .unwrap_or_default();

        let postings = if delta_postings.is_empty() {
            main_postings
        } else if main_postings.is_empty() {
            delta_postings
        } else {
            merge_sorted_lines(&main_postings, &delta_postings)
        };

        if postings.is_empty() {
            None
        } else {
            Some(postings)
        }
    }

    /// Get merged (main + delta) sorted line postings for a trigram hash.
    fn trigram_line_postings(&self, hash: u32, ci: bool) -> Option<Vec<(u32, u32, u32)>> {
        let main = self.main_posting_data(hash, ci);
        let delta = self.delta_posting_data(hash, ci);

        if main.is_none() && delta.is_none() {
            return None;
        }

        let main_postings = main
            .map(|d| self.extract_line_postings(d))
            .unwrap_or_default();
        let delta_postings = delta
            .map(|d| self.extract_line_postings(d))
            .unwrap_or_default();

        let postings = if delta_postings.is_empty() {
            main_postings
        } else if main_postings.is_empty() {
            delta_postings
        } else {
            merge_sorted_lines(&main_postings, &delta_postings)
        };

        if postings.is_empty() {
            None
        } else {
            Some(postings)
        }
    }

    // --- Search methods ---

    pub fn search_timed(&self, pattern: &str) -> (SearchResult<'_>, SearchTiming) {
        // Patterns with inline `(?i)` (or any flag group enabling
        // case-insensitive matching) can't be answered from the
        // case-sensitive store — `(?i)abc` looking up the trigram "abc"
        // would miss files containing `ABC`. If a case-insensitive
        // companion index is loaded we resolve against it (folding the
        // query literals the same way the content was folded); otherwise
        // we fall back to scanning every live file.
        let pattern_ci = trigram::has_case_insensitive_flag(pattern);
        let ci = pattern_ci && self.has_ci();
        if pattern_ci && !ci {
            let docs = self.live_doc_ids();
            let n = docs.len();
            return (
                SearchResult::AllFiles(docs),
                SearchTiming {
                    lookup_ms: 0.0,
                    bitmap_intersect_ms: 0.0,
                    verify_ms: 0.0,
                    candidates: n,
                    matches: 0,
                    strategy: String::new(),
                    density: 0.0,
                    prefix_filtered: 0,
                },
            );
        }

        let alternatives = if ci {
            trigram::decompose_pattern_folded(pattern)
        } else {
            trigram::decompose_pattern(pattern)
        };

        if alternatives.is_empty() || alternatives.iter().all(|a| a.is_empty()) {
            let docs = self.live_doc_ids();
            let n = docs.len();
            return (
                SearchResult::AllFiles(docs),
                SearchTiming {
                    lookup_ms: 0.0,
                    bitmap_intersect_ms: 0.0,
                    verify_ms: 0.0,
                    candidates: n,
                    matches: 0,
                    strategy: String::new(),
                    density: 0.0,
                    prefix_filtered: 0,
                },
            );
        }

        let mut result_lines: Vec<(u32, u32, u32)> = Vec::new();
        let mut bitmap_dur = Duration::ZERO;
        let mut intersect_dur = Duration::ZERO;
        let has_bitmaps = if ci {
            self.bitmap_ci_mmap.is_some()
        } else {
            self.bitmap_mmap.is_some()
        };

        for alt_trigrams in &alternatives {
            if alt_trigrams.is_empty() {
                let docs = self.live_doc_ids();
                let n = docs.len();
                return (
                    SearchResult::AllFiles(docs),
                    SearchTiming {
                        lookup_ms: 0.0,
                        bitmap_intersect_ms: 0.0,
                        verify_ms: 0.0,
                        candidates: n,
                        matches: 0,
                        strategy: String::new(),
                        density: 0.0,
                        prefix_filtered: 0,
                    },
                );
            }

            let hashes: Vec<u32> = alt_trigrams
                .iter()
                .map(|tri| crc32fast::hash(tri))
                .collect();

            if has_bitmaps {
                // === Two-tier search: Roaring Bitmap → filtered postings ===

                // Phase 1: Bitmap intersection (parallel load, then serial AND)
                let t_bitmap = Instant::now();

                // Parallel deserialization of all bitmaps
                let mut bitmaps: Vec<Option<RoaringBitmap>> = hashes
                    .par_iter()
                    .map(|&h| self.lookup_bitmap(h, ci))
                    .collect();

                // If any trigram is missing from bitmaps, fall back to full posting list search
                if bitmaps.iter().any(|b| b.is_none()) {
                    bitmap_dur += t_bitmap.elapsed();
                    // Fallback: load postings directly without bitmap pre-filter
                    let t_fallback = Instant::now();
                    let posting_lists: Vec<Option<Vec<(u32, u32, u32)>>> = hashes
                        .par_iter()
                        .map(|&h| self.trigram_line_postings(h, ci))
                        .collect();
                    if posting_lists.iter().any(|p| p.is_none()) {
                        intersect_dur += t_fallback.elapsed();
                        continue;
                    }
                    let mut posting_lists: Vec<Vec<(u32, u32, u32)>> =
                        posting_lists.into_iter().map(|p| p.unwrap()).collect();
                    posting_lists.sort_by_key(|v| v.len());
                    let mut candidates = posting_lists.swap_remove(0);
                    for other in &posting_lists {
                        if candidates.is_empty() {
                            break;
                        }
                        candidates = sorted_intersect_lines(&candidates, other);
                    }
                    intersect_dur += t_fallback.elapsed();
                    result_lines.extend(candidates);
                    continue;
                }

                // Sort by cardinality (smallest first) for faster AND
                bitmaps.sort_by_key(|b| b.as_ref().map_or(0, |bm| bm.len()));

                // Sequential AND with early termination
                let mut candidate_docs = bitmaps.swap_remove(0).unwrap();
                for bm in bitmaps.into_iter().flatten() {
                    candidate_docs &= bm;
                    if candidate_docs.is_empty() {
                        break;
                    }
                }

                // Delta docs don't have bitmap entries — add all live delta
                // doc_ids so incremental updates are never invisible.
                let main_count = self.main_num_docs as u32;
                for delta_idx in 0..self.delta_doc_ids.len() as u32 {
                    let doc_id = main_count + delta_idx;
                    if !self.deleted_docs.contains(&doc_id) {
                        candidate_docs.insert(doc_id);
                    }
                }

                if candidate_docs.is_empty() {
                    bitmap_dur += t_bitmap.elapsed();
                    continue;
                }

                let bm_card = candidate_docs.len() as usize;
                bitmap_dur += t_bitmap.elapsed();

                // Fast path: if bitmap is very selective (< 0.7% of corpus, with a 500-doc
                // floor so tiny corpora still benefit) AND there's only one alternative —
                // skip posting load and verify files directly.
                // NOTE: with multiple alternatives (alternation like a|b|c), we must NOT
                // return early here because other alternatives still need to be processed.
                let bitmap_threshold = ((self.num_docs() as f64 * 0.007) as usize).max(500);
                if bm_card <= bitmap_threshold && alternatives.len() == 1 {
                    let paths: Vec<&Path> = candidate_docs
                        .iter()
                        .filter_map(|id| self.doc_path(id))
                        .collect();
                    let n = paths.len();
                    return (
                        SearchResult::BitmapFiles(paths),
                        SearchTiming {
                            lookup_ms: bitmap_dur.as_secs_f64() * 1000.0,
                            bitmap_intersect_ms: 0.0,
                            verify_ms: 0.0,
                            candidates: n,
                            matches: 0,
                            strategy: String::new(),
                            density: 0.0,
                            prefix_filtered: 0,
                        },
                    );
                }

                // Phase 2: Load line postings (filtered by bitmap if selective), then intersect
                let t_intersect = Instant::now();
                let bm_selectivity = bm_card as f64 / self.num_docs().max(1) as f64;

                let posting_lists: Vec<Option<Vec<(u32, u32, u32)>>> = if bm_selectivity < 0.5 {
                    // Bitmap is selective — use filtered extraction
                    hashes
                        .par_iter()
                        .map(|&h| self.trigram_line_postings_filtered(h, &candidate_docs, ci))
                        .collect()
                } else {
                    // Bitmap isn't selective — full extraction is faster
                    hashes
                        .par_iter()
                        .map(|&h| self.trigram_line_postings(h, ci))
                        .collect()
                };

                if posting_lists.iter().any(|p| p.is_none()) {
                    intersect_dur += t_intersect.elapsed();
                    continue;
                }

                let mut posting_lists: Vec<Vec<(u32, u32, u32)>> =
                    posting_lists.into_iter().map(|p| p.unwrap()).collect();
                posting_lists.sort_by_key(|v| v.len());

                let mut candidates = posting_lists.swap_remove(0);
                for other in &posting_lists {
                    if candidates.is_empty() {
                        break;
                    }
                    candidates = sorted_intersect_lines(&candidates, other);
                }
                intersect_dur += t_intersect.elapsed();
                result_lines.extend(candidates);
            } else {
                // === Fallback: original sorted merge approach (no bitmap files) ===

                let t_lookup = Instant::now();
                let posting_lists: Vec<Option<Vec<(u32, u32, u32)>>> = hashes
                    .par_iter()
                    .map(|&h| self.trigram_line_postings(h, ci))
                    .collect();
                bitmap_dur += t_lookup.elapsed();

                if posting_lists.iter().any(|p| p.is_none()) {
                    continue;
                }

                let t_intersect = Instant::now();
                let mut posting_lists: Vec<Vec<(u32, u32, u32)>> =
                    posting_lists.into_iter().map(|p| p.unwrap()).collect();
                posting_lists.sort_by_key(|v| v.len());

                let mut candidates = posting_lists.swap_remove(0);
                for other in &posting_lists {
                    if candidates.is_empty() {
                        break;
                    }
                    candidates = sorted_intersect_lines(&candidates, other);
                }
                intersect_dur += t_intersect.elapsed();
                result_lines.extend(candidates);
            }
        }

        // Dedup result lines on (doc_id, line_no)
        result_lines.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        result_lines.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let num_candidates = result_lines.len();
        let hits: Vec<LineHit<'_>> = result_lines
            .iter()
            .filter_map(|&(doc_id, line_no, byte_offset)| {
                self.doc_path(doc_id).map(|path| LineHit {
                    path,
                    line_no,
                    byte_offset,
                })
            })
            .collect();

        (
            SearchResult::LineHits(hits),
            SearchTiming {
                lookup_ms: bitmap_dur.as_secs_f64() * 1000.0,
                bitmap_intersect_ms: intersect_dur.as_secs_f64() * 1000.0,
                verify_ms: 0.0,
                candidates: num_candidates,
                matches: 0,
                strategy: String::new(),
                density: 0.0,
                prefix_filtered: 0,
            },
        )
    }

    /// Return paths for all non-deleted docs (fallback for short patterns).
    fn live_doc_ids(&self) -> Vec<&Path> {
        (0..self.num_docs() as u32)
            .filter(|id| !self.deleted_docs.contains(id))
            .filter_map(|id| self.doc_path(id))
            .collect()
    }

    pub fn is_stale(&self) -> bool {
        // Phase 1: Check directory mtimes (no walk needed).
        // Detects file additions, deletions, and renames.
        for (path_str, &stored_mtime) in &self.meta.dir_mtimes {
            let path = Path::new(path_str);
            match fs::metadata(path) {
                Ok(m) => {
                    if mtime_secs(&m) != stored_mtime {
                        return true;
                    }
                }
                Err(_) => return true, // directory was removed
            }
        }
        // Phase 2: Sample file mtimes to detect content modifications.
        for (path_str, &stored_mtime) in self.meta.file_mtimes.iter().take(100) {
            let path = Path::new(path_str);
            match fs::metadata(path) {
                Ok(m) => {
                    if mtime_secs(&m) != stored_mtime {
                        return true;
                    }
                }
                Err(_) => return true,
            }
        }
        false
    }
}

/// Serialize one trigram map to its six on-disk files under `output`, named
/// `{prefix}.postings`, `{prefix}.lookup`, `{prefix}.bitmaps`,
/// `{prefix}.bitmaps.lookup`, `{prefix}.masks`, `{prefix}.masks.lookup`.
/// Used for both the case-sensitive map (`prefix = "ngrams"`) and the
/// case-insensitive companion (`prefix = "ngrams.ci"`). Returns the
/// postings byte length.
///
/// `masks` is the corpus-level successor mask built alongside `ngrams` in
/// the indexing pass. Each mask entry is a fixed 256-bit (8 × u32) bitmap
/// written verbatim in the order of the trigram list, so the lookup table
/// can be shared with the postings lookup (same hash → same row).
///
/// Public so the streaming build's single-chunk fast path can re-use the
/// legacy serializer instead of spilling + k-way-merging.
pub fn write_ngram_files(
    output: &Path,
    prefix: &str,
    ngrams: &HashMap<[u8; 3], TrigramBuilder>,
    masks: &HashMap<[u8; 3], FollowerMask>,
) -> Result<u64> {
    // Sort trigrams by hash for binary search
    let mut trigram_list: Vec<(&[u8; 3], &TrigramBuilder)> = ngrams.iter().collect();
    trigram_list.sort_by_key(|(k, _)| crc32fast::hash(*k));

    // Write compact (delta-varint) postings and build lookup (offset/len per
    // trigram). Postings are already compact-encoded in `builder.bytes`, so we
    // write them verbatim — no re-encode pass.
    let postings_path = output.join(format!("{prefix}.postings"));
    let mut postings_file = BufWriter::new(File::create(&postings_path)?);
    let mut lookup_entries: Vec<(u32, u64, u32)> = Vec::new();
    let mut offset: u64 = 0;
    for (tri, builder) in &trigram_list {
        let len = builder.bytes.len() as u32;
        postings_file.write_all(&builder.bytes)?;
        lookup_entries.push((crc32fast::hash(*tri), offset, len));
        offset += len as u64;
    }
    postings_file.flush()?;
    let postings_len = offset;

    let lookup_path = output.join(format!("{prefix}.lookup"));
    let mut lookup_file = BufWriter::new(File::create(&lookup_path)?);
    for (hash, off, len) in &lookup_entries {
        lookup_file.write_u32::<LittleEndian>(*hash)?;
        lookup_file.write_u64::<LittleEndian>(*off)?;
        lookup_file.write_u32::<LittleEndian>(*len)?;
    }
    lookup_file.flush()?;

    // Write Roaring Bitmaps (Tier 1: doc_id sets per trigram)
    let bitmaps_path = output.join(format!("{prefix}.bitmaps"));
    let mut bitmaps_file = BufWriter::new(File::create(&bitmaps_path)?);
    let mut bitmap_lookup_entries: Vec<(u32, u64, u32)> = Vec::new();
    let mut bm_offset: u64 = 0;
    for (tri, builder) in &trigram_list {
        let mut bitmap = RoaringBitmap::new();
        for (doc_id, _, _) in PostingReader::new(&builder.bytes) {
            bitmap.insert(doc_id);
        }
        let mut bm_buf = Vec::new();
        bitmap.serialize_into(&mut bm_buf)?;
        let bm_len = bm_buf.len() as u32;
        bitmaps_file.write_all(&bm_buf)?;
        bitmap_lookup_entries.push((crc32fast::hash(*tri), bm_offset, bm_len));
        bm_offset += bm_len as u64;
    }
    bitmaps_file.flush()?;

    let bitmaps_lookup_path = output.join(format!("{prefix}.bitmaps.lookup"));
    let mut bm_lookup_file = BufWriter::new(File::create(&bitmaps_lookup_path)?);
    for (hash, off, len) in &bitmap_lookup_entries {
        bm_lookup_file.write_u32::<LittleEndian>(*hash)?;
        bm_lookup_file.write_u64::<LittleEndian>(*off)?;
        bm_lookup_file.write_u32::<LittleEndian>(*len)?;
    }
    bm_lookup_file.flush()?;

    // Write successor masks (Tier 0.5: 4-byte pre-filter). The mask is a
    // fixed 32-byte (8 × u32 LE) entry per trigram in the same hash order
    // as the postings. Trigrams with no recorded follower (no 4th byte in
    // any file) are still emitted as a zero mask so the binary search
    // stays aligned with the trigram list — this means a missing follower
    // is represented as "all bits zero", which mask_overlap will report
    // as "no overlap" with any non-empty allowed set. The lookup table
    // mirrors the postings lookup (same hash) for symmetry with the rest
    // of the format; a more compact representation could be devised if
    // 32 bytes/trigram ever dominates the on-disk footprint.
    let masks_path = output.join(format!("{prefix}.masks"));
    let mut masks_file = BufWriter::new(File::create(&masks_path)?);
    let mut mask_lookup_entries: Vec<(u32, u64, u32)> = Vec::new();
    let mut mask_offset: u64 = 0;
    let zero_mask: FollowerMask = [0u32; 8];
    for (tri, _) in &trigram_list {
        let mask = masks.get(*tri).unwrap_or(&zero_mask);
        for word in mask {
            masks_file.write_u32::<LittleEndian>(*word)?;
        }
        mask_lookup_entries.push((crc32fast::hash(*tri), mask_offset, MASK_ENTRY_SIZE as u32));
        mask_offset += MASK_ENTRY_SIZE as u64;
    }
    masks_file.flush()?;

    let masks_lookup_path = output.join(format!("{prefix}.masks.lookup"));
    let mut mask_lookup_file = BufWriter::new(File::create(&masks_lookup_path)?);
    for (hash, off, len) in &mask_lookup_entries {
        mask_lookup_file.write_u32::<LittleEndian>(*hash)?;
        mask_lookup_file.write_u64::<LittleEndian>(*off)?;
        mask_lookup_file.write_u32::<LittleEndian>(*len)?;
    }
    mask_lookup_file.flush()?;

    Ok(postings_len)
}

/// Remove a case-insensitive companion index's files (used when (re)building a
/// case-sensitive-only index over a directory that previously had a CI index).
fn remove_ci_files(output: &Path) {
    for suffix in [
        "ngrams.ci.postings",
        "ngrams.ci.lookup",
        "ngrams.ci.bitmaps",
        "ngrams.ci.bitmaps.lookup",
        "ngrams.ci.masks",
        "ngrams.ci.masks.lookup",
        "delta.ci.postings",
        "delta.ci.lookup",
    ] {
        let _ = fs::remove_file(output.join(suffix));
    }
}

pub fn build(
    root: &Path,
    output: &Path,
    no_ignore: bool,
    type_filter: &[String],
    verbose: bool,
    case_insensitive: bool,
    build_config: Option<crate::build::StreamingConfig>,
) -> Result<()> {
    if verbose {
        eprintln!("Building index for {:?}...", root);
    }

    fs::create_dir_all(output).context("creating output directory")?;

    // Caller-supplied overrides win; otherwise default with `verbose` set.
    let cfg = build_config
        .unwrap_or_else(|| crate::build::StreamingConfig::defaults_with_verbose(verbose));
    let paths = crate::build::collect_paths(root, output, no_ignore, type_filter)?;
    let result = crate::build::streaming_build_from_paths(&paths, case_insensitive, &cfg, output)?;

    // Collect per-file mtimes keyed by the canonical string form. Same scheme
    // the legacy path used (see comment at `write_index_files`).
    let mut file_mtimes = HashMap::new();
    for path in &result.indexed_paths {
        if let Ok(m) = fs::metadata(path) {
            file_mtimes.insert(path.to_string_lossy().into_owned(), mtime_secs(&m));
        }
    }

    let dir_mtimes = collect_dir_mtimes(root, no_ignore, Some(output));

    write_docids_and_meta(
        output,
        &result.indexed_paths,
        result.num_ngrams,
        root,
        file_mtimes,
        dir_mtimes,
        case_insensitive,
    )?;

    // Clean up any delta files and stale lock from previous runs
    let _ = fs::remove_file(output.join("lock"));

    if verbose {
        eprintln!(
            "Index built: {} docs, {} trigrams, postings {}KB{}",
            result.num_docs,
            result.num_ngrams,
            result.postings_len / 1024,
            if case_insensitive {
                " (+ case-insensitive index)"
            } else {
                ""
            }
        );
    }

    Ok(())
}

/// Write `docids.bin` and `meta.json` to `output` for a streaming-built
/// index. The streaming path emits `ngrams.*` (and `ngrams.ci.*` when
/// applicable) directly, so this helper handles only the bookkeeping files
/// plus the stale-file cleanup. Shared between `persist::build` and
/// `persist::compact` so the on-disk layout stays consistent regardless of
/// how the trigram store was produced.
fn write_docids_and_meta(
    output: &Path,
    paths: &[PathBuf],
    num_ngrams: usize,
    root: &Path,
    file_mtimes: HashMap<String, u64>,
    dir_mtimes: HashMap<String, u64>,
    case_insensitive: bool,
) -> Result<()> {
    // Write docids.bin (same format as the legacy path: u16 length prefix
    // then UTF-8 bytes for each path, in the same order as indexed).
    let docids_path = output.join("docids.bin");
    {
        let mut docids_file = BufWriter::new(File::create(&docids_path)?);
        for path in paths {
            let path_bytes = path.to_string_lossy();
            let bytes = path_bytes.as_bytes();
            docids_file.write_u16::<LittleEndian>(bytes.len() as u16)?;
            docids_file.write_all(bytes)?;
        }
        docids_file.flush()?;
    }

    let meta = IndexMeta {
        version: INDEX_VERSION,
        num_docs: paths.len(),
        num_ngrams,
        root_dir: root.to_string_lossy().into_owned(),
        built_at: chrono_now(),
        file_mtimes,
        dir_mtimes,
        main_num_docs: Some(paths.len()),
        case_insensitive,
    };
    let meta_path = output.join("meta.json");
    let meta_json = serde_json::to_string_pretty(&meta)?;
    fs::write(&meta_path, meta_json)?;

    // If the corpus used to be CI and is now CS, drop the now-stale
    // ngrams.ci.* files so the index dir describes its actual mode.
    if !case_insensitive {
        remove_ci_files(output);
    }

    // Old bitmap files are overwritten by the new ones above; delta files
    // and lock are cleared so the directory describes a clean main index.
    let _ = fs::remove_file(output.join("delta.postings"));
    let _ = fs::remove_file(output.join("delta.lookup"));
    let _ = fs::remove_file(output.join("delta.docids"));
    let _ = fs::remove_file(output.join("deleted.bin"));
    let _ = fs::remove_file(output.join("delta.ci.postings"));
    let _ = fs::remove_file(output.join("delta.ci.lookup"));

    Ok(())
}

/// Serialize a fully-built `SparseIndex` to disk under `output`, writing the
/// trigram store, optional CI companion, docids table, and `meta.json`. Also
/// removes any stale delta + lock files from a prior incremental run so the
/// directory describes a single, self-contained main index.
#[allow(dead_code)] // legacy fallback; the streaming path uses `write_docids_and_meta` + `merge_to_store` directly.
fn write_index_files(
    output: &Path,
    index: &SparseIndex,
    root: &Path,
    file_mtimes: HashMap<String, u64>,
    dir_mtimes: HashMap<String, u64>,
) -> Result<u64> {
    // Write the case-sensitive trigram files, and the case-insensitive
    // companion (`ngrams.ci.*`) when this is a CI build.
    let postings_len = write_ngram_files(output, "ngrams", &index.ngrams, &index.masks)?;
    if let Some(ci) = &index.ngrams_ci {
        // The CI mask is paired with the CI trigram map. `masks_ci` is
        // always built lockstep with `ngrams_ci` inside `add_document`, so
        // it must be `Some` here; the static empty map is only a fallback
        // for defensive symmetry with the CS path.
        static EMPTY: std::sync::OnceLock<HashMap<[u8; 3], crate::index::FollowerMask>> =
            std::sync::OnceLock::new();
        let ci_masks = index
            .masks_ci
            .as_ref()
            .unwrap_or_else(|| EMPTY.get_or_init(HashMap::new));
        write_ngram_files(output, "ngrams.ci", ci, ci_masks)?;
    } else {
        // A prior CI build over this directory may have left stale CI files.
        remove_ci_files(output);
    }

    // Write docids
    let docids_path = output.join("docids.bin");
    let mut docids_file = BufWriter::new(File::create(&docids_path)?);
    for path in &index.doc_ids {
        let path_bytes = path.to_string_lossy();
        let bytes = path_bytes.as_bytes();
        docids_file.write_u16::<LittleEndian>(bytes.len() as u16)?;
        docids_file.write_all(bytes)?;
    }
    docids_file.flush()?;

    // Write meta
    let num_docs = index.doc_ids.len();
    let meta = IndexMeta {
        version: INDEX_VERSION,
        num_docs,
        num_ngrams: index.ngrams.len(),
        root_dir: root.to_string_lossy().into_owned(),
        built_at: chrono_now(),
        file_mtimes,
        dir_mtimes,
        main_num_docs: Some(num_docs),
        case_insensitive: index.ngrams_ci.is_some(),
    };
    let meta_path = output.join("meta.json");
    let meta_json = serde_json::to_string_pretty(&meta)?;
    fs::write(&meta_path, meta_json)?;

    // Old bitmap files are overwritten by the new ones above; delta files
    // and lock are cleared so the directory describes a clean main index.
    let _ = fs::remove_file(output.join("delta.postings"));
    let _ = fs::remove_file(output.join("delta.lookup"));
    let _ = fs::remove_file(output.join("delta.docids"));
    let _ = fs::remove_file(output.join("deleted.bin"));
    let _ = fs::remove_file(output.join("delta.ci.postings"));
    let _ = fs::remove_file(output.join("delta.ci.lookup"));

    Ok(postings_len)
}

/// True if an index exists at `idx_path` and is the current on-disk format
/// version. Callers use this to decide whether to (re)build before searching or
/// updating — a missing OR stale-version index returns `false`.
pub fn is_current(idx_path: &Path) -> bool {
    match fs::read_to_string(idx_path.join("meta.json")) {
        Ok(s) => serde_json::from_str::<IndexMeta>(&s)
            .map(|m| m.version == INDEX_VERSION)
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Read a delta lookup + postings pair (used for both the CS and CI deltas).
/// Returns empty vecs when the lookup file is absent.
fn read_delta(lookup_path: &Path, postings_path: &Path) -> Result<(Vec<LookupEntry>, Vec<u8>)> {
    if !lookup_path.exists() {
        return Ok((Vec::new(), Vec::new()));
    }
    let ldata = fs::read(lookup_path)?;
    let num = ldata.len() / LOOKUP_ENTRY_SIZE;
    let mut dlookup = Vec::with_capacity(num);
    let mut cursor = std::io::Cursor::new(&ldata);
    for _ in 0..num {
        let hash = cursor.read_u32::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        let len = cursor.read_u32::<LittleEndian>()?;
        dlookup.push(LookupEntry { hash, offset, len });
    }
    let dpostings = fs::read(postings_path).unwrap_or_default();
    Ok((dlookup, dpostings))
}

/// Write a delta store's `{postings, lookup}` files (compact-encoded), or
/// remove them when the trigram map is empty. Doc-ids are shared between the CS
/// and CI deltas, so this writes only the postings + lookup pair.
fn write_delta_store(
    postings_path: &Path,
    lookup_path: &Path,
    ngrams: HashMap<u32, Vec<Posting>>,
) -> Result<()> {
    let mut sorted: Vec<(u32, Vec<Posting>)> = ngrams.into_iter().collect();
    sorted.sort_by_key(|(hash, _)| *hash);

    if sorted.is_empty() {
        let _ = fs::remove_file(postings_path);
        let _ = fs::remove_file(lookup_path);
        return Ok(());
    }

    let mut postings_file = BufWriter::new(File::create(postings_path)?);
    let mut lookup_entries: Vec<(u32, u64, u32)> = Vec::new();
    let mut offset: u64 = 0;
    for (hash, postings) in &sorted {
        let mut buf = Vec::with_capacity(postings.len() * 3);
        let mut w = PostingWriter::new();
        for &(doc_id, line_no, byte_offset) in postings {
            w.push(&mut buf, doc_id, line_no, byte_offset);
        }
        let len = buf.len() as u32;
        postings_file.write_all(&buf)?;
        lookup_entries.push((*hash, offset, len));
        offset += len as u64;
    }
    postings_file.flush()?;

    let mut lookup_file = BufWriter::new(File::create(lookup_path)?);
    for (hash, off, len) in &lookup_entries {
        lookup_file.write_u32::<LittleEndian>(*hash)?;
        lookup_file.write_u64::<LittleEndian>(*off)?;
        lookup_file.write_u32::<LittleEndian>(*len)?;
    }
    lookup_file.flush()?;
    Ok(())
}

/// Open the eight mmaps of a trigram store given its file prefix, returning
/// `None` for the whole set when the postings file is absent. The mask
/// files are optional (older indices pre-date them); the corresponding
/// `Option` fields are `None` when the files are missing.
type StoreMmaps = (
    Mmap,
    usize,
    Mmap,
    Option<Mmap>,
    Option<Mmap>,
    usize,
    Option<Mmap>,
    Option<Mmap>,
    usize,
);
fn load_store(index_path: &Path, prefix: &str) -> Result<Option<StoreMmaps>> {
    let postings_path = index_path.join(format!("{prefix}.postings"));
    if !postings_path.exists() {
        return Ok(None);
    }
    let lf = File::open(index_path.join(format!("{prefix}.lookup")))?;
    let lookup_mmap = unsafe { Mmap::map(&lf)? };
    let lookup_count = lookup_mmap.len() / LOOKUP_ENTRY_SIZE;
    let pf = File::open(&postings_path)?;
    let postings_mmap = unsafe { Mmap::map(&pf)? };

    let bitmaps_path = index_path.join(format!("{prefix}.bitmaps"));
    let bitmaps_lookup_path = index_path.join(format!("{prefix}.bitmaps.lookup"));
    let (bitmap_mmap, bitmap_lookup_mmap, bitmap_lookup_count) =
        if bitmaps_path.exists() && bitmaps_lookup_path.exists() {
            let bf = File::open(&bitmaps_path)?;
            let bm = unsafe { Mmap::map(&bf)? };
            let blf = File::open(&bitmaps_lookup_path)?;
            let blm = unsafe { Mmap::map(&blf)? };
            let count = blm.len() / LOOKUP_ENTRY_SIZE;
            (Some(bm), Some(blm), count)
        } else {
            (None, None, 0)
        };

    // Successor-mask files: optional. When absent, mask_overlap returns
    // true for every query (no pruning), so the searcher still works
    // correctly on indexes built by older versions of `fgr`.
    let masks_path = index_path.join(format!("{prefix}.masks"));
    let masks_lookup_path = index_path.join(format!("{prefix}.masks.lookup"));
    let (masks_mmap, masks_lookup_mmap, masks_count) =
        if masks_path.exists() && masks_lookup_path.exists() {
            let mf = File::open(&masks_path)?;
            let mm = unsafe { Mmap::map(&mf)? };
            let mlf = File::open(&masks_lookup_path)?;
            let mlm = unsafe { Mmap::map(&mlf)? };
            let count = mlm.len() / LOOKUP_ENTRY_SIZE;
            (Some(mm), Some(mlm), count)
        } else {
            (None, None, 0)
        };

    Ok(Some((
        lookup_mmap,
        lookup_count,
        postings_mmap,
        bitmap_mmap,
        bitmap_lookup_mmap,
        bitmap_lookup_count,
        masks_mmap,
        masks_lookup_mmap,
        masks_count,
    )))
}

pub fn load(index_path: &Path) -> Result<PersistentIndex> {
    let meta_path = index_path.join("meta.json");
    let meta_str = fs::read_to_string(&meta_path).context("reading meta.json")?;
    let meta: IndexMeta = serde_json::from_str(&meta_str).context("parsing meta.json")?;

    if meta.version != INDEX_VERSION {
        // Auto-migrate: an older index format can't be read by the v4
        // delta-varint decoder, so rebuild from the root the previous index
        // covered. The build() call reuses the recorded root_dir and
        // case_insensitive flag, preserving the user's intent (the build is
        // the migration — there's no per-format converter worth maintaining).
        // We refuse to migrate when the meta has no root_dir, which can
        // happen on partial/broken indices, and ask the caller to rebuild.
        if meta.root_dir.is_empty() {
            anyhow::bail!(
                "index at {} is version {} but this build expects version {} and has no root_dir recorded; rebuild with `fgr index`",
                index_path.display(),
                meta.version,
                INDEX_VERSION
            );
        }
        eprintln!(
            "[fgr] Migrating index at {} from version {} to {} (rebuilding)",
            index_path.display(),
            meta.version,
            INDEX_VERSION
        );
        let root = Path::new(&meta.root_dir);
        let ci = meta.case_insensitive;
        // Clear the index directory before the rebuild so the directory walk
        // (which may not be filtered by .gitignore on every system) doesn't
        // index its own .postings/.lookup files as if they were sources.
        let _ = fs::remove_dir_all(index_path);
        build(root, index_path, false, &[], false, ci, None)
            .context("auto-migrating older index")?;
        // Recurse into the freshly-built v4 index. Recursion is bounded:
        // build() writes the current INDEX_VERSION, so the recursive call
        // passes the version check and returns immediately.
        return load(index_path);
    }

    // Load main lookup via mmap (zero-copy binary search)
    let lookup_path = index_path.join("ngrams.lookup");
    let lookup_file = File::open(&lookup_path).context("opening ngrams.lookup")?;
    let lookup_mmap = unsafe { Mmap::map(&lookup_file)? };
    let lookup_count = lookup_mmap.len() / LOOKUP_ENTRY_SIZE;

    // Load main postings mmap
    let postings_path = index_path.join("ngrams.postings");
    let postings_file = File::open(&postings_path).context("opening ngrams.postings")?;
    let postings_mmap = unsafe { Mmap::map(&postings_file)? };

    // Load Roaring Bitmap files (optional — backward compatible with older indexes)
    let bitmaps_path = index_path.join("ngrams.bitmaps");
    let bitmaps_lookup_path = index_path.join("ngrams.bitmaps.lookup");
    let (bitmap_mmap, bitmap_lookup_mmap, bitmap_lookup_count) =
        if bitmaps_path.exists() && bitmaps_lookup_path.exists() {
            let bf = File::open(&bitmaps_path).context("opening ngrams.bitmaps")?;
            let bm = unsafe { Mmap::map(&bf)? };
            let blf = File::open(&bitmaps_lookup_path).context("opening ngrams.bitmaps.lookup")?;
            let blm = unsafe { Mmap::map(&blf)? };
            let count = blm.len() / LOOKUP_ENTRY_SIZE;
            (Some(bm), Some(blm), count)
        } else {
            (None, None, 0)
        };

    // Load case-sensitive successor mask files (optional — older indexes
    // pre-date the mask). When absent, mask_overlap returns true (no
    // pruning) and the search falls back to its existing two-tier path.
    let masks_path = index_path.join("ngrams.masks");
    let masks_lookup_path = index_path.join("ngrams.masks.lookup");
    let (masks_mmap, masks_lookup_mmap, masks_count) =
        if masks_path.exists() && masks_lookup_path.exists() {
            let mf = File::open(&masks_path).context("opening ngrams.masks")?;
            let mm = unsafe { Mmap::map(&mf)? };
            let mlf = File::open(&masks_lookup_path).context("opening ngrams.masks.lookup")?;
            let mlm = unsafe { Mmap::map(&mlf)? };
            let count = mlm.len() / LOOKUP_ENTRY_SIZE;
            (Some(mm), Some(mlm), count)
        } else {
            (None, None, 0)
        };

    // Load main doc_ids via mmap (zero-alloc offset table)
    let docids_path = index_path.join("docids.bin");
    let docids_file = File::open(&docids_path).context("opening docids.bin")?;
    let docids_mmap = unsafe { Mmap::map(&docids_file)? };
    let mut docid_offsets = Vec::new();
    {
        let data = &*docids_mmap;
        let mut pos = 0usize;
        while pos + 2 <= data.len() {
            let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if pos + len > data.len() {
                break;
            }
            docid_offsets.push((pos as u32, len as u16));
            pos += len;
        }
    }

    let main_num_docs = meta.main_num_docs.unwrap_or(docid_offsets.len());

    // Load deleted set (if exists)
    let deleted_path = index_path.join("deleted.bin");
    let deleted_docs = if deleted_path.exists() {
        let data = fs::read(&deleted_path)?;
        let mut set = HashSet::new();
        let mut cursor = std::io::Cursor::new(&data);
        while (cursor.position() as usize) + 4 <= data.len() {
            if let Ok(id) = cursor.read_u32::<LittleEndian>() {
                set.insert(id);
            }
        }
        set
    } else {
        HashSet::new()
    };

    // Load delta index (if exists)
    let delta_docids_path = index_path.join("delta.docids");
    let (delta_lookup, delta_postings) = read_delta(
        &index_path.join("delta.lookup"),
        &index_path.join("delta.postings"),
    )?;

    // Load the case-insensitive companion store and its delta (if present).
    let (
        lookup_ci_mmap,
        lookup_ci_count,
        postings_ci_mmap,
        bitmap_ci_mmap,
        bitmap_lookup_ci_mmap,
        bitmap_lookup_ci_count,
        masks_ci_mmap,
        masks_ci_lookup_mmap,
        masks_ci_count,
    ) = match if meta.case_insensitive {
        load_store(index_path, "ngrams.ci")?
    } else {
        None
    } {
        Some((lm, lc, pm, bm, blm, bc, mm, mlm, mc)) => {
            (Some(lm), lc, Some(pm), bm, blm, bc, mm, mlm, mc)
        }
        None => (None, 0, None, None, None, 0, None, None, 0),
    };
    let (delta_lookup_ci, delta_postings_ci) = read_delta(
        &index_path.join("delta.ci.lookup"),
        &index_path.join("delta.ci.postings"),
    )?;

    // Load delta doc_ids (small count, keep as PathBuf)
    let mut delta_doc_ids = Vec::new();
    if delta_docids_path.exists() {
        let ddata = fs::read(&delta_docids_path)?;
        let mut cursor = std::io::Cursor::new(&ddata);
        while (cursor.position() as usize) < ddata.len() {
            let len = cursor.read_u16::<LittleEndian>()? as usize;
            let pos = cursor.position() as usize;
            if pos + len > ddata.len() {
                break;
            }
            let path_str = std::str::from_utf8(&ddata[pos..pos + len])?;
            delta_doc_ids.push(PathBuf::from(path_str));
            cursor.set_position((pos + len) as u64);
        }
    }

    Ok(PersistentIndex {
        lookup_mmap,
        lookup_count,
        postings_mmap,
        bitmap_mmap,
        bitmap_lookup_mmap,
        bitmap_lookup_count,
        docids_mmap,
        docid_offsets,
        delta_doc_ids,
        meta,
        deleted_docs,
        delta_lookup,
        delta_postings,
        main_num_docs,
        lookup_ci_mmap,
        lookup_ci_count,
        postings_ci_mmap,
        bitmap_ci_mmap,
        bitmap_lookup_ci_mmap,
        bitmap_lookup_ci_count,
        delta_lookup_ci,
        delta_postings_ci,
        masks_mmap,
        masks_lookup_mmap,
        masks_count,
        masks_ci_mmap,
        masks_ci_lookup_mmap,
        masks_ci_count,
    })
}

pub struct UpdateStats {
    pub added: usize,
    pub modified: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub duration_ms: u64,
}

pub struct CompactStats {
    pub before_main: usize,
    pub before_delta: usize,
    pub deleted_reclaimed: usize,
    pub after_total: usize,
    pub duration_ms: u64,
}

/// Merge the in-memory delta (delta_postings + delta_lookup + delta_docids +
/// deleted_docs) back into a fresh main index file. After a successful compact
/// the index directory describes a self-contained main index with no delta
/// overlay; `meta.main_num_docs` becomes the new doc count.
///
/// `verbose` prints indexing progress.
///
/// Use this after a long sequence of incremental updates so the search hot
/// path (which has to filter out deleted docs and merge with delta on every
/// query) is back to a single-mmap lookup.
pub fn compact(index_path: &Path, verbose: bool) -> Result<CompactStats> {
    let start = Instant::now();

    // 1. Load the persistent index to learn the merged doc_ids, root_dir, and
    //    whether a case-insensitive companion is in use.
    let mut pidx = load(index_path)?;

    let root_dir = pidx.meta.root_dir.clone();
    let ci_enabled = pidx.has_ci();
    let before_main = pidx.main_num_docs;
    let before_delta = pidx.num_docs() - pidx.main_num_docs;

    // 2. Effective doc_ids = main (minus deleted) ∪ delta. Main docs come
    //    first so any future incremental update appends to a stable range.
    let mut effective_paths: Vec<PathBuf> = Vec::with_capacity(pidx.num_docs());
    let mut deleted_reclaimed = 0usize;
    for id in 0..pidx.main_num_docs as u32 {
        if pidx.deleted_docs.contains(&id) {
            deleted_reclaimed += 1;
            continue;
        }
        if let Some(path) = pidx.doc_path(id) {
            effective_paths.push(path.to_path_buf());
        }
    }
    for id in pidx.main_num_docs as u32..pidx.num_docs() as u32 {
        if let Some(path) = pidx.doc_path(id) {
            effective_paths.push(path.to_path_buf());
        }
    }

    // Release the mmaps before rewriting the index files. Windows refuses to
    // truncate a file that still has an active mapping (os error 1224).
    pidx.close();
    drop(pidx);

    // 3. Acquire the index lock so a concurrent `update_incremental` can't
    //    clobber our rewrite of the main files. The lock is released on drop
    //    of the file handle at end of scope.
    let (_lock, _waited) =
        acquire_index_lock(index_path).context("acquiring index lock for compact")?;

    if verbose {
        eprintln!(
            "Compacting index at {}: {} main + {} delta → rewriting as fresh main index",
            index_path.display(),
            before_main,
            before_delta,
        );
    }

    // 4. Reindex the effective file list with the streaming path
    //    (skips the directory walk — we already know exactly which files
    //    belong to the corpus).
    let cfg = crate::build::StreamingConfig::defaults_with_verbose(verbose);
    let result =
        crate::build::streaming_build_from_paths(&effective_paths, ci_enabled, &cfg, index_path)?;

    // 5. Refresh mtime caches: every file we're now indexing should carry its
    //    current mtime, and we don't trust stale ones.
    let mut file_mtimes = HashMap::with_capacity(result.indexed_paths.len());
    for path in &result.indexed_paths {
        if let Ok(m) = fs::metadata(path) {
            file_mtimes.insert(path.to_string_lossy().into_owned(), mtime_secs(&m));
        }
    }
    let dir_mtimes = collect_dir_mtimes(Path::new(&root_dir), false, Some(index_path));

    // 6. Write docids.bin + meta.json + cleanup (the streaming path already
    //    produced ngrams.* and ngrams.ci.* directly into `index_path`).
    let root = Path::new(&root_dir);
    write_docids_and_meta(
        index_path,
        &result.indexed_paths,
        result.num_ngrams,
        root,
        file_mtimes,
        dir_mtimes,
        ci_enabled,
    )?;

    let after_total = result.num_docs as usize;

    if verbose {
        eprintln!(
            "Compact done: {} docs (was {} + {} delta, {} deleted reclaimed) in {}ms",
            after_total,
            before_main,
            before_delta,
            deleted_reclaimed,
            start.elapsed().as_millis(),
        );
    }

    Ok(CompactStats {
        before_main,
        before_delta,
        deleted_reclaimed,
        after_total,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

pub fn update_incremental(index_path: &Path, root: &Path, verbose: bool) -> Result<UpdateStats> {
    let start = Instant::now();

    // 1. Load meta.json — get saved file_mtimes
    let meta_path = index_path.join("meta.json");
    let meta_str = fs::read_to_string(&meta_path).context("reading meta.json")?;
    let meta: IndexMeta = serde_json::from_str(&meta_str).context("parsing meta.json")?;
    // An incremental update only rewrites the delta; it cannot mix a new-format
    // delta into an old-format main index without corrupting it. Rebuild from
    // scratch instead — `build()` writes the current version, after which
    // the caller can re-run the update.
    if meta.version != INDEX_VERSION {
        if meta.root_dir.is_empty() {
            anyhow::bail!(
                "index at {} is version {} but this build expects version {} and has no root_dir recorded; rebuild with `fgr index`",
                index_path.display(),
                meta.version,
                INDEX_VERSION
            );
        }
        eprintln!(
            "[fgr] Index version mismatch on update; rebuilding from {:?} before continuing",
            meta.root_dir
        );
        let root = Path::new(&meta.root_dir);
        let ci = meta.case_insensitive;
        // See the matching note in load() — wipe the index dir first so the
        // rebuild walk doesn't index its own files.
        let _ = fs::remove_dir_all(index_path);
        build(root, index_path, false, &[], false, ci, None)
            .context("auto-migrating older index before update")?;
        // After the rebuild there's nothing to incrementally update — the
        // rebuild already re-walked the whole tree. Return an empty stats
        // record so the caller knows no delta was applied.
        return Ok(UpdateStats {
            added: 0,
            modified: 0,
            deleted: 0,
            unchanged: meta.num_docs,
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }
    let saved_mtimes = meta.file_mtimes;
    let main_num_docs = meta.main_num_docs.unwrap_or(meta.num_docs);

    // 2. Walk root — get current file mtimes and directory mtimes
    let walker = WalkBuilder::new(root)
        .git_ignore(true)
        .hidden(false)
        .build();
    let mut current_files: HashMap<String, u64> = HashMap::new();
    let mut new_dir_mtimes: HashMap<String, u64> = HashMap::new();
    for entry in walker.flatten() {
        if entry.file_type().is_some_and(|ft| ft.is_dir()) {
            // Exclude the index directory itself to avoid self-invalidation
            if !entry.path().starts_with(index_path) {
                if let Ok(m) = entry.metadata() {
                    new_dir_mtimes
                        .insert(entry.path().to_string_lossy().into_owned(), mtime_secs(&m));
                }
            }
            continue;
        }
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        // Skip files inside the index directory
        if path.starts_with(index_path) {
            continue;
        }
        if let Ok(m) = fs::metadata(path) {
            current_files.insert(path.to_string_lossy().into_owned(), mtime_secs(&m));
        }
    }

    // 3. Classify: added, modified, deleted (vs last known state)
    let mut added_set: HashSet<String> = HashSet::new();
    let mut modified_set: HashSet<String> = HashSet::new();
    for (path, &mtime) in &current_files {
        match saved_mtimes.get(path) {
            None => {
                added_set.insert(path.clone());
            }
            Some(&saved) if saved != mtime => {
                modified_set.insert(path.clone());
            }
            _ => {}
        }
    }
    let mut deleted_set: HashSet<String> = HashSet::new();
    for path in saved_mtimes.keys() {
        if !current_files.contains_key(path) {
            deleted_set.insert(path.clone());
        }
    }

    // 4. Early return if no changes
    if added_set.is_empty() && modified_set.is_empty() && deleted_set.is_empty() {
        return Ok(UpdateStats {
            added: 0,
            modified: 0,
            deleted: 0,
            unchanged: saved_mtimes.len(),
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    // 5. Load existing index to get doc_ids and current delta/deleted state
    let pidx = load(index_path)?;

    // Build path -> doc_id mapping
    let path_to_docid: HashMap<String, u32> = (0..pidx.num_docs() as u32)
        .filter_map(|id| {
            pidx.doc_path(id)
                .map(|p| (p.to_string_lossy().into_owned(), id))
        })
        .collect();

    // 6. Update deleted set: mark deleted/modified docs
    let mut new_deleted: HashSet<u32> = pidx.deleted_docs.clone();
    for path in deleted_set.iter().chain(modified_set.iter()) {
        if let Some(&doc_id) = path_to_docid.get(path) {
            new_deleted.insert(doc_id);
        }
    }

    // 7. Determine which files go in the new delta.
    let mut delta_files_to_index: Vec<String> = Vec::new();

    // Keep existing delta files that haven't changed
    for id in main_num_docs..pidx.num_docs() {
        if new_deleted.contains(&(id as u32)) {
            continue;
        }
        if let Some(p) = pidx.doc_path(id as u32) {
            delta_files_to_index.push(p.to_string_lossy().into_owned());
        }
    }

    // Add newly added/modified files
    delta_files_to_index.extend(added_set.iter().cloned());
    delta_files_to_index.extend(modified_set.iter().cloned());

    // Drop the loaded index (releases mmap)
    drop(pidx);

    // 8. Index all delta files with line-level postings. When the index has a
    // case-insensitive companion, build the CI delta in lockstep (same delta
    // docs, case-folded trigrams) so `-i` searches stay correct after updates.
    let ci_enabled = meta.case_insensitive;
    let mut delta_ngrams: HashMap<u32, Vec<Posting>> = HashMap::new();
    let mut delta_ngrams_ci: HashMap<u32, Vec<Posting>> = HashMap::new();
    let mut fold_buf: Vec<u8> = Vec::new();
    let mut seen_on_line_ci: HashSet<[u8; 3]> = HashSet::new();
    let mut delta_doc_ids: Vec<PathBuf> = Vec::new();
    let mut actual_added = 0usize;
    let mut actual_modified = 0usize;

    for path_str in &delta_files_to_index {
        let path = Path::new(path_str);
        let content = match fs::read(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // Skip binary files
        if content.iter().take(512).any(|&b| b == 0) {
            continue;
        }

        // Doc_id in combined space: main_num_docs + delta_doc_ids.len()
        let doc_id = (main_num_docs + delta_doc_ids.len()) as u32;
        delta_doc_ids.push(path.to_path_buf());

        if added_set.contains(path_str) {
            actual_added += 1;
        } else if modified_set.contains(path_str) {
            actual_modified += 1;
        }

        if content.len() < 3 {
            continue;
        }

        // Line-level trigram indexing (same as SparseIndex::add_document)
        let mut line_no = 1u32;
        let mut line_start = 0usize;
        let mut seen_on_line: HashSet<[u8; 3]> = HashSet::new();

        loop {
            let line_end = content[line_start..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|p| line_start + p)
                .unwrap_or(content.len());

            let line = &content[line_start..line_end];
            if line.len() >= 3 {
                seen_on_line.clear();
                let byte_offset = line_start as u32;
                for w in line.windows(3) {
                    let tri = [w[0], w[1], w[2]];
                    if seen_on_line.insert(tri) {
                        let hash = crc32fast::hash(&tri);
                        delta_ngrams
                            .entry(hash)
                            .or_default()
                            .push((doc_id, line_no, byte_offset));
                    }
                }

                // Lockstep CI delta: same posting, folded trigrams.
                if ci_enabled {
                    casefold::fold_into(line, &mut fold_buf);
                    if fold_buf.len() >= 3 {
                        seen_on_line_ci.clear();
                        for w in fold_buf.windows(3) {
                            let tri = [w[0], w[1], w[2]];
                            if seen_on_line_ci.insert(tri) {
                                let hash = crc32fast::hash(&tri);
                                delta_ngrams_ci.entry(hash).or_default().push((
                                    doc_id,
                                    line_no,
                                    byte_offset,
                                ));
                            }
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
    }

    // 9. Write delta files (small -- only changed files)

    // Write deleted.bin: only main doc_ids that are deleted
    let deleted_path = index_path.join("deleted.bin");
    let main_deleted: Vec<u32> = new_deleted
        .iter()
        .filter(|&&id| (id as usize) < main_num_docs)
        .copied()
        .collect();
    if main_deleted.is_empty() {
        let _ = fs::remove_file(&deleted_path);
    } else {
        let mut f = BufWriter::new(File::create(&deleted_path)?);
        for id in &main_deleted {
            f.write_u32::<LittleEndian>(*id)?;
        }
        f.flush()?;
    }

    // Write delta postings + lookup (12 bytes per entry)
    let mut sorted_ngrams: Vec<(u32, Vec<Posting>)> = delta_ngrams.into_iter().collect();
    sorted_ngrams.sort_by_key(|(hash, _)| *hash);

    let delta_postings_path = index_path.join("delta.postings");
    let delta_lookup_path = index_path.join("delta.lookup");
    let delta_docids_path = index_path.join("delta.docids");

    if sorted_ngrams.is_empty() && delta_doc_ids.is_empty() {
        let _ = fs::remove_file(&delta_postings_path);
        let _ = fs::remove_file(&delta_lookup_path);
        let _ = fs::remove_file(&delta_docids_path);
    } else {
        let mut postings_file = BufWriter::new(File::create(&delta_postings_path)?);
        let mut lookup_entries: Vec<(u32, u64, u32)> = Vec::new();
        let mut offset: u64 = 0;

        for (hash, postings) in &sorted_ngrams {
            let mut buf = Vec::with_capacity(postings.len() * 3);
            let mut w = PostingWriter::new();
            for &(doc_id, line_no, byte_offset) in postings {
                w.push(&mut buf, doc_id, line_no, byte_offset);
            }
            let len = buf.len() as u32;
            postings_file.write_all(&buf)?;
            lookup_entries.push((*hash, offset, len));
            offset += len as u64;
        }
        postings_file.flush()?;

        let mut lookup_file = BufWriter::new(File::create(&delta_lookup_path)?);
        for (hash, off, len) in &lookup_entries {
            lookup_file.write_u32::<LittleEndian>(*hash)?;
            lookup_file.write_u64::<LittleEndian>(*off)?;
            lookup_file.write_u32::<LittleEndian>(*len)?;
        }
        lookup_file.flush()?;

        // Write delta docids
        let mut docids_file = BufWriter::new(File::create(&delta_docids_path)?);
        for path in &delta_doc_ids {
            let path_bytes = path.to_string_lossy();
            let bytes = path_bytes.as_bytes();
            docids_file.write_u16::<LittleEndian>(bytes.len() as u16)?;
            docids_file.write_all(bytes)?;
        }
        docids_file.flush()?;
    }

    // 9b. Write the lockstep CI delta (postings + lookup only — docids are
    // shared with the CS delta written above). When the index has no CI
    // companion, make sure no stale CI delta lingers.
    if ci_enabled {
        write_delta_store(
            &index_path.join("delta.ci.postings"),
            &index_path.join("delta.ci.lookup"),
            delta_ngrams_ci,
        )?;
    } else {
        let _ = fs::remove_file(index_path.join("delta.ci.postings"));
        let _ = fs::remove_file(index_path.join("delta.ci.lookup"));
    }

    // 10. Update meta.json with current file_mtimes
    let mut new_mtimes: HashMap<String, u64> = HashMap::with_capacity(saved_mtimes.len());
    for (path, &mtime) in &saved_mtimes {
        if !deleted_set.contains(path) && !modified_set.contains(path) {
            new_mtimes.insert(path.clone(), mtime);
        }
    }
    for path_str in added_set.iter().chain(modified_set.iter()) {
        if let Some(&mtime) = current_files.get(path_str) {
            if delta_doc_ids
                .iter()
                .any(|p| p.to_string_lossy() == *path_str)
            {
                new_mtimes.insert(path_str.clone(), mtime);
            }
        }
    }

    let total_docs = main_num_docs - main_deleted.len() + delta_doc_ids.len();
    let new_meta = IndexMeta {
        version: INDEX_VERSION,
        num_docs: total_docs,
        num_ngrams: meta.num_ngrams,
        root_dir: root.to_string_lossy().into_owned(),
        built_at: chrono_now(),
        file_mtimes: new_mtimes,
        dir_mtimes: new_dir_mtimes,
        main_num_docs: Some(main_num_docs),
        case_insensitive: meta.case_insensitive,
    };
    let meta_json = serde_json::to_string_pretty(&new_meta)?;
    fs::write(&meta_path, meta_json)?;

    let unchanged = total_docs - actual_added - actual_modified;

    if verbose {
        eprintln!(
            "Changes: +{} added, {} modified, {} deleted",
            actual_added,
            actual_modified,
            deleted_set.len()
        );
    }

    Ok(UpdateStats {
        added: actual_added,
        modified: actual_modified,
        deleted: deleted_set.len(),
        unchanged,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

/// Extract mtime from filesystem metadata, truncated to 2-second granularity.
/// This avoids false stale detection from sub-second timestamp jitter on NTFS.
fn mtime_secs(meta: &std::fs::Metadata) -> u64 {
    let secs = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs / 2 * 2
}

/// Walk a directory tree and collect mtime for each directory (not files).
/// Uses the `ignore` crate to respect .gitignore rules.
/// Excludes `exclude_dir` (the index directory itself) to avoid self-invalidation.
fn collect_dir_mtimes(
    root: &Path,
    no_ignore: bool,
    exclude_dir: Option<&Path>,
) -> HashMap<String, u64> {
    let mut dir_mtimes = HashMap::new();
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(!no_ignore)
        .build();
    for entry in walker.filter_map(|e| e.ok()) {
        if entry.file_type().map_or(false, |ft| ft.is_dir()) {
            if let Some(excl) = exclude_dir {
                if entry.path().starts_with(excl) {
                    continue;
                }
            }
            if let Ok(m) = entry.metadata() {
                dir_mtimes.insert(entry.path().to_string_lossy().into_owned(), mtime_secs(&m));
            }
        }
    }
    dir_mtimes
}

fn chrono_now() -> String {
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s_since_epoch", dur.as_secs())
}

/// Full staleness check: walks ALL files and directories, comparing every mtime
/// against the index metadata. Expensive (~850ms for 79K files) but zero false
/// negatives. Used by the daemon at startup.
pub fn full_stale_check(index: &PersistentIndex, index_path: &Path) -> bool {
    let root = Path::new(&index.meta.root_dir);
    let walker = ignore::WalkBuilder::new(root)
        .git_ignore(true)
        .hidden(false)
        .build();
    for entry in walker.flatten() {
        let path = entry.path();
        if path.starts_with(index_path) {
            continue;
        }
        if entry.file_type().is_some_and(|ft| ft.is_dir()) {
            let key = path.to_string_lossy();
            if let Some(&stored) = index.meta.dir_mtimes.get(key.as_ref()) {
                if let Ok(m) = entry.metadata() {
                    if mtime_secs(&m) != stored {
                        return true;
                    }
                }
            } else {
                return true; // new directory
            }
        } else if entry.file_type().is_some_and(|ft| ft.is_file()) {
            let key = path.to_string_lossy();
            if let Some(&stored) = index.meta.file_mtimes.get(key.as_ref()) {
                if let Ok(m) = fs::metadata(path) {
                    if mtime_secs(&m) != stored {
                        return true;
                    }
                }
            } else {
                return true; // new file
            }
        }
    }
    // Also check for deleted files (in index but not on disk)
    for path_str in index.meta.file_mtimes.keys() {
        if !Path::new(path_str).exists() {
            return true;
        }
    }
    false
}

/// Acquire an exclusive lock on the index directory for updates.
/// Returns the lock file handle and a flag indicating whether we had to wait.
pub fn acquire_index_lock(idx_path: &Path) -> anyhow::Result<(fs::File, bool)> {
    let lock_path = idx_path.join("lock");
    let mut waited = false;
    loop {
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(f) => return Ok((f, waited)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if !waited {
                    eprintln!("Waiting for another process to finish updating index...");
                    waited = true;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => return Err(e.into()),
        }
    }
}

pub fn release_index_lock(idx_path: &Path) {
    let _ = fs::remove_file(idx_path.join("lock"));
}
