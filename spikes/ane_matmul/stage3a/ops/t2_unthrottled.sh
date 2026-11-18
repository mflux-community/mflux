#!/bin/zsh
# Re-run the t2 concurrency sweep from an unthrottled foreground session.
# The sweep itself (t2_splits.py) builds missing models, holds BENCH.lock, and
# wraps every run in watchdog.sh. A macmon log covers the sweep as a validity
# canary: ANE power near 9 W means full rate; 2.5-5 W means session throttling.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=$(cd "$HERE/.." && pwd)
PY=$STAGE/../../../.venv/bin/python
cd "$STAGE"
mkdir -p /tmp/ane3a/models
echo "== throttle probe"
$PY ../gen_models.py /tmp/ane3a/models 4224 20480 3840 --variant c1 >/dev/null 2>&1 || true
PROBE=$(../watchdog.sh ../bench/anebench /tmp/ane3a/models/c1_4224x20480x3840.mlpackage --units ane --shape 4224 20480 3840 --layout conv --iters 30 2>/dev/null | tail -1)
echo "$PROBE"
MED=$(echo "$PROBE" | $PY -c "import json,sys; print(json.loads(sys.stdin.readline())['median_ms'])")
awk -v m="$MED" 'BEGIN { exit (m < 50 ? 0 : 1) }' || { echo "SESSION THROTTLED (median $MED ms) - stopping per the standing rule"; exit 1; }
echo "== gate (max 10 min)"
"$HERE/gate.sh" 600 || { echo "GATE FAIL - stopping per the standing rule"; exit 1; }
if [ -f results/t2_splits.jsonl ] && [ ! -f results/t2_splits_throttled.jsonl ]; then
  mv results/t2_splits.jsonl results/t2_splits_throttled.jsonl
  echo "== moved the throttled sweep to results/t2_splits_throttled.jsonl"
fi
macmon pipe -i 2000 > /tmp/ane3a/t2_macmon.jsonl 2>/dev/null &
MCPID=$!
echo "== t2 unthrottled sweep start $(date +%H:%M:%S)"
$PY -u t2_splits.py 2>&1 | grep -v -E "Torch version|SyntaxWarning|n_bits|passes/s"
st=$pipestatus[1]
kill $MCPID 2>/dev/null || true
echo "== t2 sweep exit $st $(date +%H:%M:%S)"
exit $st
