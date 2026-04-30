#!/usr/bin/env bash
# Compare two bench JSON outputs from scripts/bench.sh.
#
# Usage: scripts/bench-diff.sh BASELINE.json OPTIMIZED.json
#
# Prints a per-pattern table with mean times and percent delta. Negative
# delta = faster (good). Significance gate: a change is flagged "*" only
# if |delta| > stddev (rough — not a t-test, but enough to filter noise).
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 BASELINE.json OPTIMIZED.json" >&2
  exit 1
fi
python3 - "$1" "$2" <<'PY'
import json, sys
base, opt = (json.load(open(p)) for p in sys.argv[1:3])
b = {r["label"]: r for r in base["results"]}
o = {r["label"]: r for r in opt["results"]}
print(f"baseline:  {base['context']['git_short']}  ({base['context']['timestamp_utc']})")
print(f"optimized: {opt['context']['git_short']}  ({opt['context']['timestamp_utc']})")
print()
print(f"{'pattern':32} {'base (ms)':>12} {'opt (ms)':>12} {'delta':>10} {'significant':>12}")
for label in sorted(set(b) | set(o)):
    if label not in b or label not in o:
        print(f"{label:32}  (only in one set)")
        continue
    bm = b[label]["mean_s"] * 1000
    om = o[label]["mean_s"] * 1000
    bs = b[label]["stddev_s"] * 1000
    ds = o[label]["stddev_s"] * 1000
    delta_ms = om - bm
    delta_pct = (delta_ms / bm) * 100 if bm else 0.0
    noise = bs + ds
    sig = "*" if abs(delta_ms) > noise else " "
    arrow = "↓" if delta_ms < 0 else ("↑" if delta_ms > 0 else "·")
    print(f"{label:32} {bm:>12.2f} {om:>12.2f} {delta_pct:>+8.1f}% {arrow}  {sig:>5}")
PY
