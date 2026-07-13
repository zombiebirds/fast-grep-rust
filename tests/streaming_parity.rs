//! Streaming-build parity test (M5).
//!
//! Builds a synthetic corpus with N thousands of randomized ASCII files,
//! then runs the streaming build twice with very different chunking
//! parameters (one chunk vs many chunks) and asserts the six final files
//! are byte-identical. If the k-way merge or the per-chunk extraction is
//! non-deterministic between runs, this test catches it.
//!
//! This exercises the same on-disk path used by `fgr index` at production
//! scale, just at a smaller corpus size. Marked `#[ignore]` because it
//! allocates hundreds of MB of temp space and takes a while — run with:
//!
//!   cargo test --release -- --ignored streaming_parity
//!
//! on a workstation, not in every CI tick.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use fast_grep::build::{streaming_build_from_paths, StreamingConfig};

/// Generate `count` files of pseudo-random ASCII text between 2 KB and 20 KB
/// each, scattered across `root`. Deterministic given the same `seed`.
fn synth_corpus(root: &std::path::Path, count: usize, seed: u64) -> Vec<PathBuf> {
    let mut paths = Vec::with_capacity(count);
    let mut rng_state = seed;
    let alphabet = b"abcdefghijklmnopqrstuvwxyz\n";
    for i in 0..count {
        // 2 KB to 20 KB per file.
        let size = 2 * 1024 + (i % 19) * 1024;
        let mut bytes = Vec::with_capacity(size);
        for _ in 0..size {
            // xorshift — deterministic, no dep.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            bytes.push(alphabet[(rng_state as usize) % alphabet.len()]);
        }
        let path = root.join(format!("file_{:05}.txt", i));
        fs::write(&path, &bytes).unwrap();
        paths.push(path);
    }
    paths
}

/// Compare two files via stable hash. Returns `None` when either is missing
/// or any read fails — the caller is responsible for asserting both sides
/// produced files.
fn file_digest(path: &std::path::Path) -> Option<u64> {
    let bytes = fs::read(path).ok()?;
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    Some(h.finish())
}

#[test]
#[ignore]
fn streaming_parity_different_chunk_sizes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let paths = synth_corpus(tmp.path(), 5000, 0xC0FFEE_u64);
    assert_eq!(paths.len(), 5000);

    // Path A: one big chunk (whole corpus into a single spill).
    let dir_a = tempfile::TempDir::new().unwrap();
    let cfg_a = StreamingConfig {
        chunk_files: 10_000,        // > corpus size → 1 chunk
        chunk_byte_target: 1 << 30, // 1 GiB (effectively unbounded at this corpus size)
        max_posting_buf_bytes: 256 * 1024 * 1024,
        write_buffer_bytes: 4 * 1024 * 1024,
        defer_first_chunk: true,
        verbose: false,
    };
    let res_a = streaming_build_from_paths(&paths, false, &cfg_a, dir_a.path()).unwrap();

    // Path B: many small chunks (force multi-spill + k-way merge).
    let dir_b = tempfile::TempDir::new().unwrap();
    let cfg_b = StreamingConfig {
        chunk_files: 128, // ~40 chunks
        chunk_byte_target: 4 * 1024 * 1024,
        max_posting_buf_bytes: 256 * 1024 * 1024,
        write_buffer_bytes: 4 * 1024 * 1024,
        defer_first_chunk: true,
        verbose: false,
    };
    let res_b = streaming_build_from_paths(&paths, false, &cfg_b, dir_b.path()).unwrap();

    // Both must report the same metadata.
    assert_eq!(res_a.num_docs, res_b.num_docs);
    assert_eq!(res_a.num_ngrams, res_b.num_ngrams);
    // Postings byte volume can differ very slightly in the rare case the
    // per-trigram PostingWriter's internal byte offsets between
    // (file_count, line_no) records land at the exact varint width
    // boundary — both encodings decode to the same (doc, line) tuples,
    // so digest equality is what really matters.
    assert_eq!(res_a.postings_len, res_b.postings_len);

    // Six-file digest equality across both outputs.
    let suffix_pairs = [
        "ngrams.postings",
        "ngrams.lookup",
        "ngrams.bitmaps",
        "ngrams.bitmaps.lookup",
        "ngrams.masks",
        "ngrams.masks.lookup",
    ];
    for suffix in suffix_pairs {
        let a_path = dir_a.path().join(suffix);
        let b_path = dir_b.path().join(suffix);
        let a = file_digest(&a_path).expect(&format!("missing file {}", a_path.display()));
        let b = file_digest(&b_path).expect(&format!("missing file {}", b_path.display()));
        assert_eq!(
            a, b,
            "digest mismatch for {} — streaming build must be deterministic across chunk sizes",
            suffix
        );
    }
}
