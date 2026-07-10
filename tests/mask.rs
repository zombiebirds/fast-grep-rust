//! Integration tests for the successor-mask (phrase-aware trigram pre-filter).
//!
//! The mask is a 256-bit bitmap per trigram, recording which bytes can
//! immediately follow the trigram anywhere in the corpus. The on-disk
//! layout is two extra files per trigram store: `{prefix}.masks` (32 bytes
//! per entry, 8 × u32 LE) and `{prefix}.masks.lookup` (16-byte hash →
//! offset/len, same shape as the postings lookup).
//!
//! These tests cover:
//!  1. Fresh builds write mask files for both CS and CI stores.
//!  2. `mask_overlap` returns the expected truth values against real
//!     followers and against provably-absent followers.
//!  3. The cross-line follower is captured (the mask is built from the
//!     whole-file stream, not per-line).
//!  4. The CI mask reflects folded content, not the original bytes.
//!  5. Backward compat: an index without mask files (e.g. one written by
//!     an older build, or one with the mask files removed by hand) loads
//!     successfully and `mask_overlap` reports "no pruning" for any
//!     query.

use std::fs;
use std::path::Path;

use fast_grep::persist::{build as build_index, load as load_index, IndexMeta};

fn write_files(tmp: &Path, files: &[(&str, &str)]) {
    for &(name, content) in files {
        fs::write(tmp.join(name), content).unwrap();
    }
}

fn mask_files_exist(idx_dir: &Path) -> bool {
    idx_dir.join("ngrams.masks").exists() && idx_dir.join("ngrams.masks.lookup").exists()
}

fn ci_mask_files_exist(idx_dir: &Path) -> bool {
    idx_dir.join("ngrams.ci.masks").exists()
        && idx_dir.join("ngrams.ci.masks.lookup").exists()
}

/// Convenience: CRC32 of a 3-byte trigram, matching the index's key hash.
fn tri_hash(tri: &[u8; 3]) -> u32 {
    crc32fast::hash(tri)
}

#[test]
fn fresh_build_writes_mask_files() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "alpha bravo charlie\n")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");
    assert!(
        mask_files_exist(&idx_dir),
        "CS mask files must be present after build"
    );
    assert!(
        !ci_mask_files_exist(&idx_dir),
        "CI mask files must NOT be present in a CS-only build"
    );
}

#[test]
fn ci_build_writes_cs_and_ci_masks() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "Alpha Bravo\n")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, true).expect("build CI");
    assert!(mask_files_exist(&idx_dir), "CS mask files present");
    assert!(ci_mask_files_exist(&idx_dir), "CI mask files present");
}

#[test]
fn mask_overlap_returns_true_for_actual_followers() {
    // "abcdefg" — the trigram "abc" is followed by 'd'.
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "abcdefg")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    let idx = load_index(&idx_dir).expect("load");
    let h = tri_hash(b"abc");
    assert!(
        idx.mask_overlap(h, b"d", false),
        "abc is followed by 'd' somewhere in the corpus"
    );
    // The mask records every observed follower, not just the one we asked
    // about — "abc" is also followed by nothing (it's only 3 bytes from
    // the start, so no 4th byte), but here the file is longer, so 'b'/'c'
    // shouldn't appear as followers either. Stick to the one true answer.
    assert!(
        !idx.mask_overlap(h, b"x", false),
        "abc is NOT followed by 'x' in this corpus"
    );
}

#[test]
fn mask_overlap_handles_multiple_candidate_bytes() {
    // "abc 123" — trigram "abc" is followed by ' ' (space). '1' is a
    // follower of trigram "bc ", not "abc", so passing a mixed allowed
    // set must still report the right answer.
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "abc 123")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    let idx = load_index(&idx_dir).expect("load");
    let h_abc = tri_hash(b"abc");
    // Any of {space, x, y, z} — only space is actually a follower.
    assert!(
        idx.mask_overlap(h_abc, b" xyz", false),
        "abc IS followed by space, so the OR over the allowed set is true"
    );
    assert!(
        !idx.mask_overlap(h_abc, b"xyz", false),
        "abc is NOT followed by any of x/y/z"
    );
    // Empty allowed set → never overlap.
    assert!(!idx.mask_overlap(h_abc, b"", false));
}

#[test]
fn mask_overlap_captures_cross_line_follower() {
    // "abc\nxyz" — trigram "bc\n" straddles the line break, and its
    // immediate successor is 'x' (first byte of the next line).
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "abc\nxyz")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    let idx = load_index(&idx_dir).expect("load");
    let h = tri_hash(b"bc\n");
    assert!(
        idx.mask_overlap(h, b"x", false),
        "cross-line successor must be recorded"
    );
}

#[test]
fn ci_mask_reflects_folded_content() {
    // "ABCdef" → folded "abcdef" (ASCII lowercase). The CI trigram "abc"
    // must exist (and so must its mask) and the mask must record 'd' as
    // a successor — NOT 'D', which only appears in the un-folded content.
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "ABCdef")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, true).expect("build CI");

    let idx = load_index(&idx_dir).expect("load CI");
    let h_abc = tri_hash(b"abc");
    assert!(
        idx.mask_overlap(h_abc, b"d", true),
        "CI mask for folded 'abc' must record folded follower 'd'"
    );
    // The un-folded capital 'D' must NOT be a recorded follower of the
    // folded trigram (casefold normalises to lowercase).
    assert!(
        !idx.mask_overlap(h_abc, b"D", true),
        "CI mask must not record un-folded 'D' as a follower of folded 'abc'"
    );
}

#[test]
fn mask_overlap_is_no_pruning_for_unknown_trigram() {
    // A trigram that doesn't exist in the corpus has no mask entry —
    // `mask_overlap` returns true (no pruning) so the searcher falls
    // through to the bitmap stage, which also finds nothing. The mask
    // is a pre-filter only, never a soundness boundary.
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "hello world")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    let idx = load_index(&idx_dir).expect("load");
    // "zzz" doesn't appear anywhere.
    let h = tri_hash(b"zzz");
    assert!(
        idx.mask_overlap(h, b"a", false),
        "mask_overlap on a missing trigram must return true (no pruning)"
    );
}

#[test]
fn backward_compat_loads_index_without_mask_files() {
    // Simulate an index written by a build that didn't emit mask files
    // (e.g. an older `fgr`). Loading must succeed and `mask_overlap`
    // must report "no pruning" for any query so the searcher still
    // returns correct results.
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "alpha beta\n")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    // Remove the mask files post-build to simulate an older index.
    fs::remove_file(idx_dir.join("ngrams.masks")).unwrap();
    fs::remove_file(idx_dir.join("ngrams.masks.lookup")).unwrap();

    // Patch the meta version to something plausible (it was the current
    // version anyway, but we don't want to drop the v4 mmap code paths).
    let meta_path = idx_dir.join("meta.json");
    let mut meta: IndexMeta =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
    meta.version = fast_grep::persist::INDEX_VERSION; // unchanged, just re-write
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).unwrap();

    let idx = load_index(&idx_dir).expect("load succeeds without mask files");
    // No mask → no pruning, for any trigram + any allowed set.
    assert!(
        idx.mask_overlap(tri_hash(b"alp"), b"x", false),
        "missing mask files must mean mask_overlap returns true"
    );

    // And the searcher still works on the unmasked index.
    let hits: Vec<_> = fast_grep::searcher::search_persistent_timed(
        &idx,
        "beta",
        None,
        false,
        &[],
        &[],
        &[],
    )
    .unwrap()
    .0
    .iter()
    .map(|m| (m.path.file_name().unwrap().to_owned(), m.line_number))
    .collect();
    assert_eq!(hits.len(), 1, "search must still return the 'beta' hit");
}
