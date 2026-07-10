use std::path::PathBuf;
use std::sync::OnceLock;

use criterion::{criterion_group, criterion_main, Criterion};

fn get_bench_dir() -> PathBuf {
    let linux = PathBuf::from("/tmp/linux-6.6");
    if linux.exists() {
        return linux;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let projects = PathBuf::from(&home).join("Projects");
    if projects.exists() {
        return projects;
    }
    PathBuf::from(".")
}

/// Locate or build the index used by the `indexed_*` benchmarks. The
/// index lives at `<bench-dir>/.fgr/index` so it can be shared across runs
/// without polluting the user's own `.fgr`. Build is skipped on later
/// runs as long as the on-disk version matches.
fn bench_index(dir: &std::path::Path) -> Option<fast_grep::persist::PersistentIndex> {
    let idx_dir = dir.join(".fgr/index");
    if !idx_dir.join("meta.json").exists() {
        eprintln!("[bench] building index under {:?}", idx_dir);
        fast_grep::persist::build(dir, &idx_dir, false, &[], false, false).ok()?;
    }
    if !fast_grep::persist::is_current(&idx_dir) {
        eprintln!("[bench] rebuilding stale index under {:?}", idx_dir);
        let _ = std::fs::remove_dir_all(&idx_dir);
        fast_grep::persist::build(dir, &idx_dir, false, &[], false, false).ok()?;
    }
    fast_grep::persist::load(&idx_dir).ok()
}

static INDEX: OnceLock<Option<fast_grep::persist::PersistentIndex>> = OnceLock::new();

fn with_index<F: FnOnce(&fast_grep::persist::PersistentIndex)>(dir: &std::path::Path, f: F) {
    let cached = INDEX.get_or_init(|| bench_index(dir));
    if let Some(idx) = cached.as_ref() {
        f(idx);
    } else {
        eprintln!("[bench] no index available; skipping indexed benchmarks");
    }
}

fn bench_patterns(c: &mut Criterion) {
    let dir = get_bench_dir();
    let patterns = [
        "EXPORT_SYMBOL",
        "static.*inline",
        "int main",
        "TODO|FIXME",
        "printk",
    ];

    let mut group = c.benchmark_group("search");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    for pattern in &patterns {
        group.bench_function(format!("full_scan/{}", pattern), |b| {
            b.iter(|| {
                fast_grep::searcher::search_full_scan(
                    &dir,
                    pattern,
                    false,
                    false,
                    &[],
                    &[],
                    &[],
                    false,
                )
                .unwrap();
            });
        });
    }

    for pattern in &patterns {
        group.bench_function(format!("indexed/{}", pattern), |b| {
            b.iter(|| {
                with_index(&dir, |idx| {
                    fast_grep::searcher::search_persistent_timed(
                        idx, pattern, None, false, &[], &[], &[],
                    )
                    .unwrap();
                });
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_patterns);
criterion_main!(benches);
