#!/bin/zsh
# Run a command; kill it if system memory use goes above 97%.
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  free=$(memory_pressure 2>/dev/null | awk '/free percentage/{gsub("%","",$5);print $5}')
  if [ -n "$free" ] && [ "$free" -lt 3 ]; then echo "WATCHDOG: memory >97%, killing $pid" >&2; kill -9 $pid; exit 137; fi
  sleep 1
done
wait $pid
