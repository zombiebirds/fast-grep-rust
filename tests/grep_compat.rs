//! End-to-end CLI tests for the grep-compatibility flags that the rest
//! of the conversation called out as "declared but ignored":
//! `-v` / `--invert-match`, `-o` / `--only-matching`, `--include GLOB`,
//! and `--exclude GLOB`.
//!
//! Goes through the actual `fgr` binary (`CARGO_BIN_EXE_fgr`) so each
//! test exercises clap parsing, SearchOpts threading, and the search
//! pipeline together — same as a real user would. That's heavier than
//! calling `searcher::*` directly but catches wiring bugs (which is
//! exactly what these flags suffered from before).
//!
//! Each test builds an isolated tempdir of fixtures, runs the binary
//! with the relevant flags, and parses its stdout. Stderr (timing
//! summary, daemon notices) is dropped.

use std::path::Path;
use std::process::Command;

fn fgr() -> Command {
    Command::new(env!("CARGO_BIN_EXE_fgr"))
}

fn write_files(tmp: &Path, files: &[(&str, &str)]) {
    for (name, content) in files {
        let full = tmp.join(name);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full, content).unwrap();
    }
}

/// Run fgr with the given args and return stdout split into trimmed lines.
fn run(args: &[&str], cwd: &Path) -> Vec<String> {
    let out = fgr()
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("spawn fgr");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect()
}

// --- --invert-match ---

#[test]
fn invert_match_returns_non_matching_lines() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("f.txt", "alpha\nbeta hit\ngamma\ndelta hit\nepsilon\n")],
    );
    let lines = run(&["-v", "hit", "f.txt"], tmp.path());
    let bodies: Vec<_> = lines
        .iter()
        .filter_map(|l| l.split(':').nth(2).map(|s| s.to_string()))
        .collect();
    assert_eq!(bodies, vec!["alpha", "gamma", "epsilon"]);
}

#[test]
fn invert_match_with_count_reports_non_matching_line_count() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("f.txt", "alpha\nbeta hit\ngamma\ndelta hit\nepsilon\n")],
    );
    let lines = run(&["-v", "-c", "hit", "f.txt"], tmp.path());
    // `path:N` per file with at least one match.
    assert_eq!(lines.len(), 1);
    assert!(lines[0].ends_with(":3"), "got {:?}", lines);
}

#[test]
fn invert_match_falls_back_when_index_is_set() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(tmp.path(), &[("f.txt", "alpha\nbeta hit\ngamma\n")]);
    // Build the index then ask for invert-match — the CLI should auto-route
    // to direct scan rather than return false negatives.
    let _ = fgr()
        .args(["index", "."])
        .current_dir(tmp.path())
        .output()
        .unwrap();
    let lines = run(&["--index", ".fgr", "-v", "hit", "f.txt"], tmp.path());
    let bodies: Vec<_> = lines
        .iter()
        .filter_map(|l| l.split(':').nth(2).map(|s| s.to_string()))
        .collect();
    assert_eq!(bodies, vec!["alpha", "gamma"]);
}

// --- --only-matching ---

#[test]
fn only_matching_emits_one_entry_per_substring() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("f.rs", "fn main() { fn helper() }\nfn another() {}\n")],
    );
    let lines = run(&["-o", "fn", "f.rs"], tmp.path());
    let bodies: Vec<_> = lines
        .iter()
        .filter_map(|l| l.split(':').nth(2).map(|s| s.to_string()))
        .collect();
    // Two `fn` on line 1, one on line 2 → 3 entries total.
    assert_eq!(bodies, vec!["fn", "fn", "fn"]);
}

#[test]
fn only_matching_extracts_regex_capture() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("f.rs", "fn main() { fn helper() }\nfn another() {}\n")],
    );
    let lines = run(&["-o", r"fn \w+", "f.rs"], tmp.path());
    let bodies: Vec<_> = lines
        .iter()
        .filter_map(|l| l.split(':').nth(2).map(|s| s.to_string()))
        .collect();
    assert_eq!(bodies, vec!["fn main", "fn helper", "fn another"]);
}

#[test]
fn only_matching_with_count_still_reports_line_counts() {
    // grep/ripgrep convention: `-c` overrides the per-substring expansion
    // that `-o` would otherwise apply, since `-c` is per-file by definition.
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("f.rs", "fn main() { fn helper() }\nfn another() {}\n")],
    );
    let lines = run(&["-o", "-c", "fn", "f.rs"], tmp.path());
    assert_eq!(lines.len(), 1);
    // Two matching lines (line 1 and line 2), regardless of three substring
    // matches across them.
    assert!(lines[0].ends_with(":2"), "got {:?}", lines);
}

// --- --include / --exclude ---

#[test]
fn include_glob_filters_files_in() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.txt", "beta hit\n"),
            ("c.md", "gamma hit\n"),
        ],
    );
    let lines = run(&["--include", "*.rs", "hit", "."], tmp.path());
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("a.rs"), "got {:?}", lines);
}

#[test]
fn exclude_glob_filters_files_out() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.txt", "beta hit\n"),
            ("c.md", "gamma hit\n"),
        ],
    );
    let lines = run(&["--exclude", "*.md", "hit", "."], tmp.path());
    assert_eq!(lines.len(), 2);
    let paths: Vec<_> = lines.iter().filter_map(|l| l.split(':').next()).collect();
    assert!(paths.iter().any(|p| p.contains("a.rs")));
    assert!(paths.iter().any(|p| p.contains("b.txt")));
    assert!(!paths.iter().any(|p| p.contains("c.md")));
}

#[test]
fn include_and_exclude_compose() {
    // `--include *.txt` would let both b.txt and notes.txt through, but
    // `--exclude notes.*` removes notes.txt — only b.txt survives.
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.txt", "beta hit\n"),
            ("notes.txt", "gamma hit\n"),
            ("c.md", "delta hit\n"),
        ],
    );
    let lines = run(
        &["--include", "*.txt", "--exclude", "notes.*", "hit", "."],
        tmp.path(),
    );
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("b.txt"), "got {:?}", lines);
}

#[test]
fn type_flag_accumulates_when_repeated() {
    // `--type rs --type ts` should match files of either extension.
    // The OR'd union mirrors how ripgrep treats repeated --type.
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.ts", "beta hit\n"),
            ("c.py", "gamma hit\n"),
        ],
    );
    let lines = run(&["--type", "rs", "--type", "ts", "hit", "."], tmp.path());
    let paths: Vec<_> = lines.iter().filter_map(|l| l.split(':').next()).collect();
    assert_eq!(lines.len(), 2, "got {:?}", lines);
    assert!(paths.iter().any(|p| p.ends_with("a.rs")));
    assert!(paths.iter().any(|p| p.ends_with("b.ts")));
    assert!(!paths.iter().any(|p| p.ends_with("c.py")));
}

#[test]
fn include_flag_accumulates_when_repeated() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.py", "beta hit\n"),
            ("c.md", "gamma hit\n"),
        ],
    );
    let lines = run(
        &["--include", "*.rs", "--include", "*.py", "hit", "."],
        tmp.path(),
    );
    assert_eq!(lines.len(), 2, "got {:?}", lines);
    let paths: Vec<_> = lines.iter().filter_map(|l| l.split(':').next()).collect();
    assert!(paths.iter().any(|p| p.ends_with("a.rs")));
    assert!(paths.iter().any(|p| p.ends_with("b.py")));
}

#[test]
fn exclude_flag_accumulates_when_repeated() {
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.md", "beta hit\n"),
            ("c.txt", "gamma hit\n"),
            ("d.py", "delta hit\n"),
        ],
    );
    let lines = run(
        &["--exclude", "*.md", "--exclude", "*.txt", "hit", "."],
        tmp.path(),
    );
    assert_eq!(lines.len(), 2, "got {:?}", lines);
    let paths: Vec<_> = lines.iter().filter_map(|l| l.split(':').next()).collect();
    assert!(paths.iter().any(|p| p.ends_with("a.rs")));
    assert!(paths.iter().any(|p| p.ends_with("d.py")));
}

#[test]
fn type_filter_works_on_indexed_path_too() {
    // --type was honoured at index-build time and on direct scan, but the
    // indexed search path silently dropped it before this fix. Build an
    // index covering all files, then search with --type and verify only
    // the matching extension surfaces.
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[
            ("a.rs", "alpha hit\n"),
            ("b.txt", "beta hit\n"),
            ("c.md", "gamma hit\n"),
        ],
    );
    let _ = fgr()
        .args(["index", "."])
        .current_dir(tmp.path())
        .output()
        .unwrap();
    let lines = run(&["--index", ".fgr", "--type", "rs", "hit", "."], tmp.path());
    assert_eq!(lines.len(), 1, "got {:?}", lines);
    assert!(lines[0].contains("a.rs"));
}

#[test]
fn include_works_on_indexed_path_too() {
    // Build an index that covers all files, then verify the per-search
    // glob filter applies post-lookup on the indexed path.
    let tmp = tempfile::tempdir().unwrap();
    write_files(
        tmp.path(),
        &[("a.rs", "alpha hit\n"), ("b.txt", "beta hit\n")],
    );
    let _ = fgr()
        .args(["index", "."])
        .current_dir(tmp.path())
        .output()
        .unwrap();
    let lines = run(
        &["--index", ".fgr", "--include", "*.rs", "hit", "."],
        tmp.path(),
    );
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("a.rs"));
}
