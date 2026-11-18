#!/bin/zsh
# Quiet-machine gate. Usage: gate.sh MAX_WAIT_S
# Quiet = no process above 40% CPU except the desktop/system whitelist.
# Exits 0 when quiet, 1 when the budget expires (caller must stop and report).
max=${1:-600}
deadline=$(( $(date +%s) + max ))
while true; do
  busy=$(ps aux -r | awk 'NR>1 && $3>40 && $11 !~ /WindowServer|kernel_task|anebench|watchdog|slow_state|fused_sep|recompile|t2_splits|qwen-code|node|macmon|ps$|awk$|gate.sh/ {print $3" "$11}' | head -3)
  if [ -z "$busy" ]; then echo "GATE OK $(date +%H:%M:%S)"; exit 0; fi
  if [ $(date +%s) -ge $deadline ]; then echo "GATE FAIL $(date +%H:%M:%S), still busy:"; echo "$busy"; exit 1; fi
  echo "$(date +%H:%M:%S) waiting:"; echo "$busy"
  sleep 20
done
