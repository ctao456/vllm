#!/usr/bin/env bash
# Intel XPU (Arc/Battlemage) performance tuning for reproducible benchmarks
# Sets GPU frequency and calls CPU setup for apples-to-apples comparison
set -euo pipefail

if ! command -v xpu-smi >/dev/null 2>&1; then
  echo "ERROR: xpu-smi not found on host" >&2
  exit 1
fi
# Detect GPUs using xpu-smi discovery --dump (requires sudo for full info)
# Format: Device ID,Device Name,...
gpu_ids=$(sudo xpu-smi discovery --dump 1 2>/dev/null | tail -n +2 | cut -d',' -f1 | sort -n | uniq || true)
if [[ -z "${gpu_ids}" ]]; then
  echo "ERROR: unable to detect GPUs via xpu-smi discovery --dump" >&2
  exit 1
fi
gpu_count=$(echo "${gpu_ids}" | wc -l | tr -d ' ')
echo "Detected ${gpu_count} GPU(s): $(echo ${gpu_ids} | tr '\n' ' ')"
freq_mhz="${1:-}"
if [[ -z "${freq_mhz}" ]]; then
  echo "Usage: sudo bash tools/setup_perf_xpu.sh <freq_mhz>" >&2
  exit 2
fi
echo ""
echo "=== Setting GPU frequencies to ${freq_mhz} MHz ==="
success_count=0
fail_count=0
for i in ${gpu_ids}; do
  echo "Setting GPU ${i} frequency to ${freq_mhz} MHz"
  if sudo xpu-smi config -d "${i}" -t 0 --frequencyrange "${freq_mhz},${freq_mhz}" 2>&1; then
    ((success_count++)) || true
  else
    echo "WARN: failed to set frequency for GPU ${i}" >&2
    ((fail_count++)) || true
  fi
done
echo "Frequency set on ${success_count} GPU(s), ${fail_count} failed/skipped"
echo ""
echo "=== CPU Performance Setup ==="
bash "$(dirname "$0")/setup_perf_cpu.sh" 2>/dev/null || echo "WARN: CPU setup skipped (run setup_perf_cpu.sh separately if needed)"
echo ""
echo "=== Verifying settings ==="
echo ""
echo "XPU status (Device ID, Name, Clock, Memory, PCIe BW):"
# Dump: 1=ID, 2=Name, 6=Clock, 16=Memory, 25=PCIe BW
sudo xpu-smi discovery --dump 1,2,6,16,25 2>/dev/null | while IFS=',' read -r id name clock mem bw; do
  if [[ "${id}" == "Device ID" ]]; then
    continue  # skip header
  fi
  # Remove quotes from name
  name="${name//\"/}"
  echo "  GPU ${id}: ${name}"
  echo "         Clock: ${clock}, Memory: ${mem}, PCIe BW: ${bw}"
done

echo ""
echo "Done"
