# profile one request (story prompt, PROF_N tokens) then mtp_bench for reference
python3 $H/profile_req.py --base $BASE --n-out ${PROF_N:-96} --no-min-tokens; sleep 30
