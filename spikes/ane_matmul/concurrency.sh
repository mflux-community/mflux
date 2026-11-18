#!/bin/zsh
# GPU (MLX bf16) + ANE (conv c1) concurrency at w1 shape. Usage: concurrency.sh MODELDIR OUTDIR "GPU_COLS ANE_COLS" ...
M=$(realpath "$1"); OUT=$(realpath "$2"); cd "$(dirname "$0")"; shift 2; T=4224 I=3840
ane(){ out=$(./watchdog.sh ./bench/anebench $M/c1_${T}x$1x${I}.mlpackage --units ane --window $2 $3 2>&1) || { echo "$out" >&2; return 1; }; echo "$out" | tail -1 | grep -o '"median_ms":[0-9.]*'; }
gpu(){ out=$(./watchdog.sh ../../.venv/bin/python mlx_loop.py $T $1 $I $2 $3 2>&1) || { echo "$out" >&2; return 1; }; echo "$out" | tail -1; }
win(){ python3 -c "import time;s=time.time()+12;print(s,s+8)"; }
for sp in "$@"; do set -- ${=sp}
  if [ ! -d $M/c1_${T}x$2x${I}.mlpackage ]; then ../../.venv/bin/python gen_models.py $M $T $2 $I --variant c1 >/dev/null || { echo "concurrency: model generation failed" >&2; exit 1; }; fi
  read s e <<< $(win)
  ane $2 $s $e > $OUT/ane.out &
  apid=$!
  gpu $1 $s $e > $OUT/gpu.out; gst=$?
  wait $apid; ast=$?
  if [ $gst -ne 0 ] || [ $ast -ne 0 ] || [ ! -s $OUT/ane.out ]; then echo "concurrency: failed (gpu status $gst, ane status $ast)" >&2; exit 1; fi
  echo "SPLIT gpu=$1 ane=$2 | gpu: $(cat $OUT/gpu.out) | ane: $(cat $OUT/ane.out)"
done
