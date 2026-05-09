#!/usr/bin/env bash
# CPU performance tuning for reproducible benchmarks
set -euo pipefail

echo "=== CPU Performance Setup ==="

echo ""
echo "Setting CPU governor to performance..."
gov_count=0
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  if [[ -f "${gov}" ]]; then
    echo performance | sudo tee "${gov}" >/dev/null
    ((gov_count++)) || true
  fi
done
echo "Set governor on ${gov_count} CPU(s)"

if [[ -f /sys/devices/system/cpu/cpu0/power/energy_perf_bias ]]; then
  echo ""
  echo "Setting energy_perf_bias to 0 (max performance)..."
  for epb_file in /sys/devices/system/cpu/cpu*/power/energy_perf_bias; do
    [[ -f "${epb_file}" ]] && echo 0 | sudo tee "${epb_file}" >/dev/null
  done
fi

echo ""
echo "=== Verifying CPU settings ==="
gov_status=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "${gov_status}" == "performance" ]]; then
  echo "  ✓ scaling_governor: ${gov_status}"
else
  echo "  ✗ scaling_governor: ${gov_status} (expected: performance)"
fi

if [[ -f /sys/devices/system/cpu/cpu0/power/energy_perf_bias ]]; then
  epb=$(cat /sys/devices/system/cpu/cpu0/power/energy_perf_bias 2>/dev/null || echo "unknown")
  if [[ "${epb}" == "0" ]]; then
    echo "  ✓ energy_perf_bias: ${epb} (max performance)"
  else
    echo "  ✗ energy_perf_bias: ${epb} (expected: 0)"
  fi
fi

echo ""
echo "Done"
