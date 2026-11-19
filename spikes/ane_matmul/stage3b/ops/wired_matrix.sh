#!/bin/zsh
# One subprocess per (config, pass): wired pages never return to the OS within a process.
cd /Users/ianscrivener/MFlux/mflux-ane-stage3
for P in 1 2 3; do
  for C in W4E4 W4E1 W4E1q8 W2E1 W2E1q8; do
    spikes/ane_matmul/watchdog.sh .venv/bin/python spikes/ane_matmul/stage3b/wired_probe.py --measure wired --config $C 2>&1 | grep '"phase": "summary"'
  done
done
spikes/ane_matmul/watchdog.sh .venv/bin/python spikes/ane_matmul/stage3b/wired_probe.py --measure latency 2>&1 | grep '"phase": "summary"'
echo "MATRIX DONE"
