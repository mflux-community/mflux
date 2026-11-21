#!/bin/zsh
# Usage: run_variant.sh MODELDIR VARIANT T O I  -> one summary line (builds model if missing)
M=$(realpath "$1"); cd "$(dirname "$0")"; v=$2 T=$3; O=$4; I=$5; f=$M/${v}_${T}x${O}x${I}.mlpackage
[ -d $f ] || ../../.venv/bin/python gen_models.py $M $T $O $I --variant $v 2>&1 | grep -E "Error" | tail -1
[ -d $f ] || exit 1
devs=$(./bench/aneplan $f | grep -v const | awk '{print $NF}' | sort | uniq -c | tr '\n' ' ')
raw=$(./watchdog.sh ./bench/anebench $f --units ane 2>&1); st=$?
if [ $st -ne 0 ]; then echo "run_variant: benchmark failed (status $st): $(echo "$raw" | tail -3)" >&2; exit $st; fi
r=$(echo "$raw" | tail -1 | grep -o '"median_ms":[0-9.]*\|"tflops":[0-9.]*' | tr '\n' ' ')
[ -n "$r" ] || { echo "run_variant: no measurements for $f" >&2; exit 1; }
echo "$v ${T}x${O}x${I} [$devs] $r"
