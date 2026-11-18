#!/bin/zsh
# Generate and benchmark ANE models for all Z-Image shapes. Output: JSON lines on stdout.
M=$(realpath "${1:?model dir}"); cd "$(dirname "$0")"
for T in 64 1152 4224 5952; do for s in "3840 3840" "10240 3840" "3840 10240"; do for v in v1 v2; do
  [ "$T" = 64 ] && [ "$s" != "3840 3840" ] && continue
  set -- $=s; f=$M/${v}_${T}x$1x$2.mlpackage
  if [ ! -d $f ]; then ../../.venv/bin/python gen_models.py $M $T $1 $2 --variant $v >/dev/null || { echo "sweep: model generation failed for $f" >&2; exit 1; }; fi
  out=$(./watchdog.sh ./bench/anebench $f --units ane --iters 50 2>&1); st=$?
  if [ $st -ne 0 ]; then echo "sweep: benchmark failed for $f (status $st): $(echo "$out" | tail -2)" >&2; exit $st; fi
  echo "$out" | tail -1
done; done; done
