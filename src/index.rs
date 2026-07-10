use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ignore::WalkBuilder;

use crate::casefold;
use crate::postenc::PostingWriter;
use crate::searcher::is_known_text_ext;

pub struct IndexStats {
    pub num_docs: usize,
    pub num_ngrams: usize,
    pub estimated_ram_bytes: usize,
    pub avg_postings_len: f64,
}

/// A posting entry: (doc_id, line_no, byte_offset).
/// - line_no: 1-based line number where this trigram appears
/// - byte_offset: byte offset of the start of that line in the file
pub type Posting = (u32, u32, u32);

/// Accumulates one trigram's posting list already in the compact
/// (delta-varint) wire format. Postings are encoded into `bytes` as they are
/// added, so the build never materializes the decoded `Vec<Posting>` for the
/// whole corpus — that is what keeps the peak RAM near the on-disk size.
pub struct TrigramBuilder {
    /// Compact-encoded postings, ready to be written to `ngrams.postings`
    /// verbatim at serialize time.
    pub bytes: Vec<u8>,
    /// Delta state (prev doc/line/offset) for `bytes`.
    writer: PostingWriter,
    /// Number of postings encoded, for `stats()` / `avg_postings_len`.
    count: u32,
}

impl Default for TrigramBuilder {
    fn default() -> Self {
        // Pre-size the posting buffer so the common-case trigram (a few
        // dozen postings) never reallocates during the build. 256 bytes
        // covers ~80 same-doc postings in the compact wire format.
        Self {
            bytes: Vec::with_capacity(256),
            writer: PostingWriter::new(),
            count: 0,
        }
    }
}

/// 256-bit "which bytes can follow this trigram" mask, packed as 8 × u32
/// little-endian. Bit `b` of word `w` (where `w = b / 32`, `bit = b % 32`)
/// is set when byte value `b` is observed to immediately follow an
/// occurrence of the owning trigram somewhere in the corpus. This is the
/// "successor mask" from Cursor's phrase-aware trigram index: it lets the
/// searcher prune alternatives whose required follower is provably absent
/// from the corpus (e.g. a regex that demands `int\d` returns zero
/// candidates in a corpus where `int` is never followed by a digit).
pub type FollowerMask = [u32; 8];

pub struct SparseIndex {
    /// Trigram → compact-encoded posting list of (doc_id, line_no, byte_offset)
    pub ngrams: HashMap<[u8; 3], TrigramBuilder>,
    /// Case-folded trigrams over the *same* documents/lines, built in the same
    /// filesystem pass when the index is case-insensitive. `None` for a plain
    /// case-sensitive index. Postings carry the original-file byte offsets, so
    /// verification still reads the un-folded line.
    pub ngrams_ci: Option<HashMap<[u8; 3], TrigramBuilder>>,
    pub doc_ids: Vec<PathBuf>,
    /// Per-trigram corpus-level follower mask (case-sensitive). Built in
    /// lockstep with `ngrams` so a 4-byte query can be answered without
    /// touching the postings.
    pub masks: HashMap<[u8; 3], FollowerMask>,
    /// Per-trigram corpus-level follower mask for the case-insensitive
    /// companion. `None` for a plain case-sensitive index. Built over the
    /// folded buffer so a `(?i)` query resolves against the same store.
    pub masks_ci: Option<HashMap<[u8; 3], FollowerMask>>,
}

#[inline]
fn set_mask_bit(mask: &mut FollowerMask, follower: u8) {
    let word = (follower >> 5) & 0x7;
    let bit = follower & 0x1F;
    mask[word as usize] |= 1u32 << bit;
}

impl SparseIndex {
    /// Create an index; when `case_insensitive` is set it also accumulates the
    /// case-folded (CI) trigram map alongside the case-sensitive one.
    #[allow(dead_code)] // exercised by the in-module unit tests; the bin target has no direct callers.
    pub fn with_case_insensitive(case_insensitive: bool) -> Self {
        SparseIndex {
            ngrams: HashMap::new(),
            ngrams_ci: if case_insensitive {
                Some(HashMap::new())
            } else {
                None
            },
            doc_ids: Vec::new(),
            masks: HashMap::new(),
            masks_ci: if case_insensitive {
                Some(HashMap::new())
            } else {
                None
            },
        }
    }

    /// Create an index pre-sized for a corpus of roughly `path_count` files.
    /// Sized to roughly `path_count * 64` entries: code corpora typically see
    /// a few hundred unique trigrams per file, but the cap stops the map from
    /// ballooning past useful on synthetic benchmarks with extreme reuse.
    pub fn with_capacity(path_count: usize, case_insensitive: bool) -> Self {
        let cap = path_count.saturating_mul(64).max(8192);
        SparseIndex {
            ngrams: HashMap::with_capacity(cap),
            ngrams_ci: if case_insensitive {
                Some(HashMap::with_capacity(cap))
            } else {
                None
            },
            doc_ids: Vec::with_capacity(path_count),
            masks: HashMap::with_capacity(cap),
            masks_ci: if case_insensitive {
                Some(HashMap::with_capacity(cap))
            } else {
                None
            },
        }
    }

    pub fn add_document(&mut self, path: &Path, content: &[u8]) {
        let doc_id = self.doc_ids.len() as u32;
        self.doc_ids.push(path.to_path_buf());

        if content.len() < 3 {
            return;
        }

        // === Successor-mask pass ===
        // We process the file as a single byte stream (not per-line) so the
        // recorded followers include the byte that follows a trigram
        // straddling a line break, e.g. `bc\n` followed by `d` of the next
        // line. This matches the "trigram → follower" semantics in the
        // Cursor phrase-aware trigram index. The mask just ORs bits, so a
        // few "extra" followers for end-of-line trigrams that don't have a
        // per-line entry in the ngrams map are harmless false positives —
        // the mask is a pre-filter, never a soundness boundary.
        for w in content.windows(4) {
            let tri = [w[0], w[1], w[2]];
            let follower = w[3];
            let mask = self.masks.entry(tri).or_insert([0u32; 8]);
            set_mask_bit(mask, follower);
        }

        // === Per-line trigram posting pass ===
        // Index trigrams per line: one posting per (trigram, doc_id, line)
        let mut line_no = 1u32;
        let mut line_start = 0usize;
        // Scratch buffer for the case-folded copy of each line, reused across
        // lines so the CI pass doesn't allocate per line.
        let mut fold_buf = Vec::new();

        loop {
            let line_end = content[line_start..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|p| line_start + p)
                .unwrap_or(content.len());

            let line = &content[line_start..line_end];

            if line.len() >= 3 {
                let byte_offset = line_start as u32;
                for w in line.windows(3) {
                    let tri = [w[0], w[1], w[2]];
                    let b = self.ngrams.entry(tri).or_default();
                    // Dedup: only one posting per (doc_id, line_no) per trigram.
                    // The windows over a line hit the same (doc, line) on every
                    // repeat of a trigram, so checking the writer's last pushed
                    // posting is enough — and lets us encode on the spot.
                    if b.writer.last_dl() != Some((doc_id, line_no)) {
                        b.writer.push(&mut b.bytes, doc_id, line_no, byte_offset);
                        b.count += 1;
                    }
                }

                // Case-insensitive map: same posting, but trigrams come from the
                // case-folded line. `byte_offset` still points at the original
                // line so verification reads un-folded text.
                if let Some(ref mut ci) = self.ngrams_ci {
                    casefold::fold_into(line, &mut fold_buf);
                    if fold_buf.len() >= 3 {
                        for w in fold_buf.windows(3) {
                            let tri = [w[0], w[1], w[2]];
                            let b = ci.entry(tri).or_default();
                            if b.writer.last_dl() != Some((doc_id, line_no)) {
                                b.writer.push(&mut b.bytes, doc_id, line_no, byte_offset);
                                b.count += 1;
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

        // === CI successor-mask pass ===
        // Mirror the CS mask over the case-folded content so a `(?i)` query
        // can be pre-filtered against the same store. We fold the whole file
        // in one pass — unlike the line-level posting extraction above — to
        // capture cross-line successors in the folded buffer. As with the CS
        // mask, extra bits for trigrams that don't appear in the per-line
        // map are harmless false positives (the mask is a pre-filter only).
        if let Some(ref mut ci_masks) = self.masks_ci {
            let mut file_fold = Vec::new();
            casefold::fold_into(content, &mut file_fold);
            if file_fold.len() >= 4 {
                for w in file_fold.windows(4) {
                    let tri = [w[0], w[1], w[2]];
                    let follower = w[3];
                    let mask = ci_masks.entry(tri).or_insert([0u32; 8]);
                    set_mask_bit(mask, follower);
                }
            }
        }
    }

    pub fn stats(&self) -> IndexStats {
        let num_docs = self.doc_ids.len();
        let num_ngrams = self.ngrams.len();
        let mut estimated_ram: usize = self
            .ngrams
            .values()
            .map(|b| 3 + b.bytes.len() + 48) // key + packed postings + overhead
            .sum();
        if let Some(ci) = &self.ngrams_ci {
            estimated_ram += ci.values().map(|b| 3 + b.bytes.len() + 48).sum::<usize>();
        }
        // Successor-mask RAM: 32 bytes per trigram for the raw 8×u32 plus
        // HashMap overhead. The mask is independent of the posting list
        // length, so it's purely a function of unique trigram count.
        estimated_ram += self.masks.len() * (32 + 48);
        if let Some(ci) = &self.masks_ci {
            estimated_ram += ci.len() * (32 + 48);
        }
        let avg_len = if num_ngrams > 0 {
            self.ngrams.values().map(|b| b.count as f64).sum::<f64>() / num_ngrams as f64
        } else {
            0.0
        };
        IndexStats {
            num_docs,
            num_ngrams,
            estimated_ram_bytes: estimated_ram,
            avg_postings_len: avg_len,
        }
    }

    /// Build an index from a pre-collected list of paths. Skips the directory
    /// walk that `build_from_directory` performs; used by `compact` to reindex
    /// from the persistent index's known doc_ids without re-scanning the tree.
    pub fn build_from_paths(
        paths: &[PathBuf],
        case_insensitive: bool,
        verbose: bool,
    ) -> Result<Self> {
        use rayon::prelude::*;

        let mut index = SparseIndex::with_capacity(paths.len(), case_insensitive);
        let mut count = 0u32;
        const CHUNK_FILES: usize = 1024;
        for chunk_start in (0..paths.len()).step_by(CHUNK_FILES) {
            let chunk_end = (chunk_start + CHUNK_FILES).min(paths.len());
            let chunk = &paths[chunk_start..chunk_end];

            let chunk_contents: Vec<Option<Vec<u8>>> = chunk
                .par_iter()
                .map(|path| -> Option<Vec<u8>> {
                    let content = std::fs::read(path).ok()?;
                    // Match `search_full_scan`'s binary-detection rule so the
                    // indexed and direct-scan paths see the same set of files.
                    if !is_known_text_ext(path) && content.iter().take(512).any(|&b| b == 0) {
                        return None;
                    }
                    Some(content)
                })
                .collect();

            for (path, content) in chunk.iter().zip(chunk_contents.into_iter()) {
                if let Some(content) = content {
                    index.add_document(path, &content);
                    count += 1;
                    if verbose && count % 10000 == 0 {
                        eprintln!("  indexed {} files...", count);
                    }
                }
            }
        }

        if verbose {
            eprintln!(
                "  indexed {} files total, {} trigrams",
                count,
                index.ngrams.len()
            );
        }

        Ok(index)
    }

    pub fn build_from_directory(
        root: &Path,
        no_ignore: bool,
        type_filter: &[String],
        verbose: bool,
        case_insensitive: bool,
    ) -> Result<Self> {
        // Phase 1: collect all file paths (serial; the walker is I/O bound
        // already and the path Vec is small relative to file contents).
        let walker = WalkBuilder::new(root)
            .git_ignore(!no_ignore)
            .hidden(false)
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

        // Phase 2: read in parallel chunks (see `build_from_paths`).
        Self::build_from_paths(&paths, case_insensitive, verbose)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn reports_correct_stats() {
        let mut idx = SparseIndex::with_case_insensitive(false);
        idx.add_document(Path::new("a.ts"), b"hello world");
        idx.add_document(Path::new("b.ts"), b"hello again");

        let stats = idx.stats();
        assert_eq!(stats.num_docs, 2);
        assert!(stats.num_ngrams > 0);
        assert!(stats.avg_postings_len > 0.0);
    }

    #[test]
    fn case_insensitive_builds_folded_map() {
        let mut idx = SparseIndex::with_case_insensitive(true);
        idx.add_document(Path::new("a.ts"), b"Hello WORLD");
        let ci = idx.ngrams_ci.as_ref().expect("ci map present");
        // The folded line "hello world" must yield the lowercase trigram "hel",
        // and the original-case "Hel" must NOT appear in the CI map.
        assert!(ci.contains_key(b"hel"));
        assert!(!ci.contains_key(b"Hel"));
        // The case-sensitive map keeps the original case.
        assert!(idx.ngrams.contains_key(b"Hel"));
    }

    /// Read one bit out of a 256-bit follower mask.
    fn mask_bit(mask: &FollowerMask, b: u8) -> bool {
        (mask[(b >> 5) as usize] >> (b & 0x1F)) & 1 == 1
    }

    #[test]
    fn successor_mask_records_following_bytes() {
        // "abcdefg" → trigrams abc/bcd/cde/def, each followed by the next byte.
        let mut idx = SparseIndex::with_case_insensitive(false);
        idx.add_document(Path::new("a.txt"), b"abcdefg");
        let m = idx.masks.get(b"abc").expect("mask for abc");
        assert!(mask_bit(m, b'd'), "'d' must be recorded as a follower of 'abc'");
        assert!(!mask_bit(m, b'e'), "'e' must NOT follow 'abc' (it's two bytes away)");
        // Trigram "def" is followed by 'g'.
        let last = idx.masks.get(b"def").expect("mask for def");
        assert!(mask_bit(last, b'g'), "'g' must follow 'def'");
    }

    #[test]
    fn end_of_content_trigram_has_no_mask_entry() {
        // "abc" (3 bytes) — there's no 4th byte, so no (trigram, follower)
        // pair can be recorded. The mask map must not contain an entry for
        // "abc" at all.
        let mut idx = SparseIndex::with_case_insensitive(false);
        idx.add_document(Path::new("a.txt"), b"abc");
        assert!(
            idx.masks.get(b"abc").is_none(),
            "trigrams with no possible follower must not appear in the mask map"
        );
    }

    #[test]
    fn successor_mask_captures_cross_line_breaks() {
        // "abc\ndef" — trigram "bc\n" straddles the line break, and its
        // successor is 'd' (first byte of the next line). The whole-file
        // stream pass must record this.
        let mut idx = SparseIndex::with_case_insensitive(false);
        idx.add_document(Path::new("a.txt"), b"abc\ndef");
        let m = idx.masks.get(b"bc\n").expect("mask for bc\\n");
        assert!(mask_bit(m, b'd'), "cross-line follower must be captured");
    }

    #[test]
    fn ci_successor_mask_built_from_folded_content() {
        // "ABCdef" → folded "abcdef". The trigram "abc" in the CI map must
        // record 'd' as its follower (the 'D' in the source folds to 'd').
        let mut idx = SparseIndex::with_case_insensitive(true);
        idx.add_document(Path::new("a.txt"), b"ABCdef");
        let ci_masks = idx.masks_ci.as_ref().expect("ci masks present");
        let m = ci_masks.get(b"abc").expect("CI mask for folded abc");
        assert!(mask_bit(m, b'd'), "CI mask must reflect folded follower");
        // The un-folded trigram "ABC" should not have a CI mask entry.
        assert!(!ci_masks.contains_key(b"ABC"));
    }
}
