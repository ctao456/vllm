#!/usr/bin/env bash
# Organize the TQ 5-category benchmark artifacts into the repo's results/ tree
# with a clean, documented layout. Safe to re-run (rsync mirror). Does NOT touch
# the live bench-results working dirs or unrelated experiments.
#
#   results/
#   ├── INDEX.md                  # what's where + provenance
#   ├── REPORT_TQ_FULL.md         # aggregated 5-category report (from aggregate_all.py)
#   ├── cat1_2_capacity_throughput/   # tq_perf (4096 cap + 4 scen) + cat2_throughput (long prefills)
#   ├── cat3_sla/                 # SLA sweep JSONs + sla_summary.csv + sla_sweep.csv
#   ├── cat4_longctx/             # long-context JSONs + longctx.csv
#   └── cat5_ruler/               # RULER results.json dirs + ruler_summary.csv
set -uo pipefail
SRC="/home/intel/models/bench-results"
DST="/home/intel/ctao/turboquant/vllm/results"
mkdir -p "$DST"

# 1) capacity + throughput (two source dirs feed cat1/cat2)
mkdir -p "$DST/cat1_2_capacity_throughput/from_tq_perf" "$DST/cat1_2_capacity_throughput/from_cat2_8192"
rsync -a --delete "$SRC/tq_perf/" "$DST/cat1_2_capacity_throughput/from_tq_perf/" 2>/dev/null || true
rsync -a --delete "$SRC/cat2_throughput/" "$DST/cat1_2_capacity_throughput/from_cat2_8192/" 2>/dev/null || true

# 2-4) sla / longctx / ruler (mirror as-is)
for c in cat3_sla cat4_longctx cat5_ruler; do
  mkdir -p "$DST/$c"
  rsync -a --delete "$SRC/$c/" "$DST/$c/" 2>/dev/null || true
done

# 5) master orchestration logs
mkdir -p "$DST/orchestration_logs"
rsync -a "$SRC/overnight_logs/" "$DST/orchestration_logs/" 2>/dev/null || true

# 6) regenerate the aggregated report
python3 /home/intel/ctao/turboquant/vllm/bench/aggregate_all.py >/dev/null 2>&1 || true
cp "$SRC/REPORT_TQ_FULL.md" "$DST/REPORT_TQ_FULL.md" 2>/dev/null || true

echo "Organized into $DST"
du -sh "$DST"
