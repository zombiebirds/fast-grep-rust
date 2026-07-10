//! Integration tests for the delta compaction API.
//!
//! `compact` merges the in-memory delta (delta_postings + delta_lookup +
//! delta_docids + deleted_docs) back into a fresh main index file. After a
//! successful compact the index directory describes a self-contained main
//! index with no delta overlay, and searches must continue to return the
//! same hits as before compaction.

use std::fs;
use std::path::{Path, PathBuf};

use fast_grep::persist::{
    build as build_index, compact as compact_index, load as load_index, update_incremental,
};
use fast_grep::searcher::{search_full_scan, search_persistent_timed, Match};

fn write_files(tmp: &Path, files: &[(&str, &str)]) {
    for &(name, content) in files {
        fs::write(tmp.join(name), content).unwrap();
    }
}

fn search_set(
    idx: &fast_grep::persist::PersistentIndex,
    pattern: &str,
) -> Vec<(PathBuf, usize)> {
    let mut v: Vec<(PathBuf, usize)> = search_persistent_timed(idx, pattern, None, false, &[], &[], &[])
        .expect("indexed search")
        .0
        .iter()
        .map(|m: &Match| (m.path.to_path_buf(), m.line_number))
        .collect();
    v.sort();
    v
}

fn full_scan_set(tmp: &Path, pattern: &str) -> Vec<(PathBuf, usize)> {
    let mut v: Vec<(PathBuf, usize)> = search_full_scan(tmp, pattern, true, false, &[], &[], &[], false)
        .expect("full scan")
        .iter()
        .map(|m| (m.path.to_path_buf(), m.line_number))
        .collect();
    v.sort();
    v
}

fn paths_exist(idx_dir: &Path) -> bool {
    idx_dir.join("delta.postings").exists()
        || idx_dir.join("delta.lookup").exists()
        || idx_dir.join("delta.docids").exists()
        || idx_dir.join("deleted.bin").exists()
}

#[test]
fn compact_clears_delta_files() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("a.txt", "alpha beta gamma\nsecond line\n"), ("b.txt", "delta echo foxtrot\n")],
    );
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    // Add a new file so the next update produces a delta.
    fs::write(tmp.path().join("c.txt"), "new content here\n").unwrap();
    update_incremental(&idx_dir, tmp.path(), false).expect("update");

    // Sanity: delta files exist before compact.
    {
        let idx_pre = load_index(&idx_dir).expect("load pre");
        assert!(idx_pre.num_docs() > idx_pre.main_num_docs, "delta should be non-empty");
        assert!(paths_exist(&idx_dir), "delta files should be present pre-compact");
    }

    let stats = compact_index(&idx_dir, false).expect("compact");
    assert_eq!(stats.before_main, 2);
    assert_eq!(stats.before_delta, 1);
    assert_eq!(stats.after_total, 3);
    assert_eq!(stats.deleted_reclaimed, 0);

    // Delta files must be gone.
    assert!(!paths_exist(&idx_dir), "delta files should be cleared after compact");

    // main_num_docs in meta should equal the post-compact total.
    let idx_post = load_index(&idx_dir).expect("load post");
    assert_eq!(idx_post.main_num_docs, idx_post.num_docs());
    assert_eq!(idx_post.main_num_docs, 3);
}

#[test]
fn compact_preserves_search_results() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.txt", "alpha beta\n"),
            ("b.txt", "beta gamma\n"),
            ("c.txt", "delta epsilon\n"),
        ],
    );
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    // Force a few updates to populate the delta.
    fs::write(tmp.path().join("d.txt"), "beta in delta\n").unwrap();
    update_incremental(&idx_dir, tmp.path(), false).expect("update 1");
    fs::write(tmp.path().join("e.txt"), "more beta content\n").unwrap();
    update_incremental(&idx_dir, tmp.path(), false).expect("update 2");

    // Pre-compact: indexed search must already match full scan.
    let pre_full = {
        let idx_pre = load_index(&idx_dir).expect("load pre");
        let pre_indexed = search_set(&idx_pre, "beta");
        let pre_full = full_scan_set(tmp.path(), "beta");
        assert_eq!(pre_indexed, pre_full, "indexed != full scan pre-compact");
        pre_full
    };

    compact_index(&idx_dir, false).expect("compact");

    let idx_post = load_index(&idx_dir).expect("load post");
    let post_indexed = search_set(&idx_post, "beta");
    assert_eq!(post_indexed, pre_full, "compact changed search results for 'beta'");

    // Multiple other patterns should also be stable across compact.
    for pat in ["alpha", "gamma", "delta", "epsilon", "content"] {
        assert_eq!(
            search_set(&idx_post, pat),
            full_scan_set(tmp.path(), pat),
            "compact broke pattern {pat:?}"
        );
    }
}

#[test]
fn compact_reclaims_deleted_docs() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.txt", "keep me\n"),
            ("doomed.txt", "doomed content\n"),
        ],
    );
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    // Delete a file and re-index: this lands the deleted doc in deleted.bin.
    fs::remove_file(tmp.path().join("doomed.txt")).unwrap();
    update_incremental(&idx_dir, tmp.path(), false).expect("update");

    {
        let idx_pre = load_index(&idx_dir).expect("load pre");
        assert!(
            idx_pre.deleted_docs.contains(&1),
            "doomed.txt should be marked deleted before compact"
        );
        assert!(
            !paths_exist(&idx_dir) || idx_pre.delta_doc_ids.is_empty(),
            "should have deleted set but no live delta"
        );
    }

    let stats = compact_index(&idx_dir, false).expect("compact");
    assert!(stats.deleted_reclaimed >= 1);
    assert_eq!(stats.after_total, 1, "only the surviving file remains");

    let idx_post = load_index(&idx_dir).expect("load post");
    assert_eq!(idx_post.main_num_docs, 1);
    assert!(idx_post.deleted_docs.is_empty(), "deleted set must be cleared");
    assert_eq!(idx_post.num_docs(), 1);

    // Searches must no longer find the deleted file's content.
    let hits = search_set(&idx_post, "doomed");
    assert!(hits.is_empty(), "deleted file's content must not appear: {hits:?}");
    // The surviving file is still searchable.
    assert_eq!(search_set(&idx_post, "keep"), full_scan_set(tmp.path(), "keep"));
}

#[test]
fn compact_on_fresh_index_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "alpha\n"), ("b.txt", "beta\n")]);
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    // No incremental update has run, so there are no delta files.
    let stats = compact_index(&idx_dir, false).expect("compact fresh");
    assert_eq!(stats.before_delta, 0);
    assert_eq!(stats.deleted_reclaimed, 0);
    assert_eq!(stats.after_total, 2);

    // Index still loads and searches correctly.
    let idx = load_index(&idx_dir).expect("load");
    assert_eq!(search_set(&idx, "alpha"), full_scan_set(tmp.path(), "alpha"));
}

#[test]
fn compact_preserves_case_insensitive_companion() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.txt", "Hello World\n"),
            ("b.txt", "GOODBYE moon\n"),
        ],
    );
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, true).expect("build CI");

    fs::write(tmp.path().join("c.txt"), "another HELLO line\n").unwrap();
    update_incremental(&idx_dir, tmp.path(), false).expect("update");

    compact_index(&idx_dir, false).expect("compact");

    let idx = load_index(&idx_dir).expect("load");
    assert!(idx.has_ci(), "CI companion should survive compact");

    // CI pattern must continue to match a full scan.
    assert_eq!(
        search_set(&idx, "(?i)hello"),
        full_scan_set(tmp.path(), "(?i)hello"),
        "CI search broken after compact"
    );
    assert_eq!(
        search_set(&idx, "(?i)goodbye"),
        full_scan_set(tmp.path(), "(?i)goodbye"),
        "CI search broken after compact"
    );
}

#[test]
fn compact_after_incremental_with_modified_file() {
    // Reproduces the typical daemon-driven sequence:
    //   1) build, 2) add new file, 3) modify an existing file (causes its
    //      previous doc to land in deleted.bin and the new content in the
    //      delta), 4) compact. After compact, only the latest content of
    //      every file is searchable.
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.txt", "first version\n"),
            ("b.txt", "stable content\n"),
        ],
    );
    let idx_dir = tmp.path().join(".fgr");
    build_index(tmp.path(), &idx_dir, true, &[], false, false).expect("build");

    // Add a brand-new file so the delta has at least one entry.
    fs::write(tmp.path().join("c.txt"), "fresh addition\n").unwrap();
    // Modify an existing file. update_incremental notices the mtime change,
    // soft-deletes the old doc, and indexes the new content into the delta.
    fs::write(tmp.path().join("a.txt"), "second version with marker PATTERN\n").unwrap();

    update_incremental(&idx_dir, tmp.path(), false).expect("update");
    {
        let idx_pre = load_index(&idx_dir).expect("load pre");
        assert!(
            idx_pre.deleted_docs.contains(&0) || idx_pre.num_docs() > idx_pre.main_num_docs,
            "expected either a deleted entry or a non-empty delta"
        );
    }

    compact_index(&idx_dir, false).expect("compact");

    let idx_post = load_index(&idx_dir).expect("load post");
    assert_eq!(
        idx_post.main_num_docs,
        idx_post.num_docs(),
        "compact should leave a clean main"
    );
    assert_eq!(
        search_set(&idx_post, "PATTERN"),
        full_scan_set(tmp.path(), "PATTERN"),
        "modified file's new content must be searchable post-compact"
    );
    // The old version's unique phrase must no longer appear.
    assert!(
        search_set(&idx_post, "first version").is_empty(),
        "old version text must be gone after compact"
    );
}