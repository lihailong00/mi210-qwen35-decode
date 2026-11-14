#!/usr/bin/env bash
# Serialize GPU use + temperature guard. Usage (host): /data/tools/mi210-opt/gpu-run.sh <tag> <command...>
# Runs <command> inside container mi210-opt under an exclusive flock. Logs temps to /data/tools/mi210-opt/logs/<tag>.temp.
set -uo pipefail
W=/data/tools/mi210-opt; TAG=$1; shift
H=$(ls -d /sys/class/drm/card1/device/hwmon/hwmon* | head -1)
maxtemp(){ m=0; for t in $H/temp1_input $H/temp2_input $H/temp3_input; do v=$(cat $t 2>/dev/null || echo 0); [ $v -gt $m ] && m=$v; done; echo $((m/1000)); }
exec 9>$W/gpu.lock; flock 9
for i in $(seq 1 120); do [ $(maxtemp) -lt 60 ] && break; sleep 2; done
( while true; do echo "$(date +%T) $(maxtemp) $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 ))" >> $W/logs/$TAG.temp; sleep 1; done ) & MON=$!
ENVS=""; for v in $(env | grep -oE "^(GGML_[A-Z0-9_]*|LLAMA_[A-Z0-9_]*|CODEX_[A-Z0-9_]*|HIP_[A-Z0-9_]*|LD_LIBRARY_PATH)=" | tr -d =); do ENVS="$ENVS -e $v"; done
docker exec $ENVS mi210-opt bash -lc "export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:\$PATH; $*" &
CMD=$!
while kill -0 $CMD 2>/dev/null; do
  if [ $(maxtemp) -ge 85 ]; then echo "TEMP ABORT $(maxtemp)C" >&2; docker exec mi210-opt pkill -f "llama-|test-backend|rocprofv3" ; fi
  sleep 1
done
wait $CMD; RC=$?
kill $MON 2>/dev/null
echo "[gpu-run $TAG] rc=$RC peak=$(sort -k2 -n $W/logs/$TAG.temp | tail -1 | cut -d" " -f2)C vram_peak=$(sort -k3 -n $W/logs/$TAG.temp | tail -1 | cut -d" " -f3)MiB" >&2
exit $RC
