# job for deploy/start.sh bench: MTP check (host side)
python3 "$(dirname "$0")/mtp_bench.py" $BASE $OUT/mtp_bench.json
