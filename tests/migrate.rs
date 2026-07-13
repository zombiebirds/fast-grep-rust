//! Integration tests for auto-migration of older index versions.
//!
//! Loading an index whose `meta.json` declares a version older than the
//! current on-disk format triggers a transparent rebuild from the recorded
//! `root_dir`. After the migration the index is searchable and `meta.json`
//! reports the current version.

use std::fs;
use std::path::Path;

use fast_grep::persist::{build as build_index, load as load_index, update_incremental, IndexMeta};

fn write_files(tmp: &Path, files: &[(&str, &str)]) {
    for &(name, content) in files {
        fs::write(tmp.join(name), content).unwrap();
    }
}

/// Build a fresh v4 index, then overwrite its meta.json with a fake older
/// version. The on-disk trigram files are left untouched so a non-migrating
/// loader would crash; the auto-migrating `load()` must rebuild instead.
fn plant_old_index(tmp: &Path, idx_dir: &Path, fake_version: u32) {
    build_index(tmp, idx_dir, true, &[], false, false, None).expect("build");
    let meta_path = idx_dir.join("meta.json");
    let mut meta: IndexMeta =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
    meta.version = fake_version;
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).unwrap();
}

#[test]
fn load_auto_migrates_old_version_index() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("a.txt", "alpha beta\n"), ("b.txt", "gamma delta\n")],
    );
    let idx_dir = tmp.path().join(".fgr");
    plant_old_index(tmp.path(), &idx_dir, 1);

    // Pre-load: meta reports version 1, not the current v4.
    {
        let meta: IndexMeta =
            serde_json::from_str(&fs::read_to_string(idx_dir.join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta.version, 1);
    }

    // load() should transparently rebuild and return a v4 index.
    let idx = load_index(&idx_dir).expect("auto-migrate on load");
    assert_eq!(idx.meta.version, fast_grep::persist::INDEX_VERSION);
    assert_eq!(idx.meta.num_docs, 2);

    // Post-load: meta now reports the current version.
    let meta: IndexMeta =
        serde_json::from_str(&fs::read_to_string(idx_dir.join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta.version, fast_grep::persist::INDEX_VERSION);

    // And the migrated index is searchable.
    let hits: Vec<_> =
        fast_grep::searcher::search_persistent_timed(&idx, "beta", None, false, &[], &[], &[])
            .unwrap()
            .0
            .iter()
            .map(|m| (m.path.file_name().unwrap().to_owned(), m.line_number))
            .collect();
    assert_eq!(hits.len(), 1, "expected one hit for 'beta'");
}

#[test]
fn load_auto_migrates_multiple_legacy_versions() {
    // Walk the index through fake versions 1, 2, and 3 — every older version
    // must migrate cleanly to v4.
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "hello world\n")]);
    let idx_dir = tmp.path().join(".fgr");
    for fake in [1u32, 2, 3] {
        plant_old_index(tmp.path(), &idx_dir, fake);
        let idx = load_index(&idx_dir).expect("auto-migrate");
        assert_eq!(idx.meta.version, fast_grep::persist::INDEX_VERSION);
        assert_eq!(idx.meta.num_docs, 1);
    }
}

#[test]
fn update_incremental_auto_migrates_old_version() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("a.txt", "alpha\n")]);
    let idx_dir = tmp.path().join(".fgr");
    plant_old_index(tmp.path(), &idx_dir, 2);

    // Add a new file before the update so there's something to detect.
    fs::write(tmp.path().join("b.txt"), "beta\n").unwrap();

    let stats = update_incremental(&idx_dir, tmp.path(), false).expect("update migrates");
    // The auto-migrate runs a full rebuild, so the update reports no
    // incremental delta. (The user's mental model: the migration IS the
    // update — a fresh index is the result.)
    assert_eq!(stats.added, 0);
    assert_eq!(stats.modified, 0);
    assert_eq!(stats.deleted, 0);

    let idx = load_index(&idx_dir).expect("load after migrate-update");
    assert_eq!(idx.meta.version, fast_grep::persist::INDEX_VERSION);
    assert_eq!(
        idx.meta.num_docs, 2,
        "both files should be in the rebuilt index"
    );
}

#[test]
fn load_fails_when_old_index_has_no_root_dir() {
    // A partially-written or broken meta.json has no root_dir to rebuild
    // from. `load()` should surface a clear error rather than guessing.
    let tmp = tempfile::tempdir().unwrap();
    let idx_dir = tmp.path().join(".fgr");
    fs::create_dir_all(&idx_dir).unwrap();
    let meta = IndexMeta {
        version: 1,
        num_docs: 0,
        num_ngrams: 0,
        root_dir: String::new(), // missing root
        built_at: String::new(),
        file_mtimes: Default::default(),
        dir_mtimes: Default::default(),
        main_num_docs: None,
        case_insensitive: false,
    };
    fs::write(
        idx_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    let err = load_index(&idx_dir)
        .err()
        .expect("must reject orphan old-version meta");
    let msg = format!("{:#}", err);
    assert!(
        msg.contains("no root_dir recorded"),
        "expected helpful error, got: {msg}"
    );
}
