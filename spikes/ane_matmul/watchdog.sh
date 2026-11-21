#!/bin/zsh
# Run a command; kill it if system memory use goes above 97%.
# zsh sets BG_NICE by default, so `"$@" &` gave every timed run nice 5 without anyone asking for it.
# Niceness changes cross-thread overlap, which is exactly what these runs measure.
unsetopt BG_NICE
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  free=$(memory_pressure 2>/dev/null | awk '/free percentage/{gsub("%","",$5);print $5}')
  if [ -n "$free" ] && [ "$free" -lt 3 ]; then echo "WATCHDOG: memory >97%, killing $pid" >&2; kill -9 $pid; exit 137; fi
  sleep 1
done
wait $pid
