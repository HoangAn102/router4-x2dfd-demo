#!/bin/bash
# E2 supervisor, part 2 (replaces e2_supervisor.sh after stage a failed its 64-frame verify, 01/11 16:47):
# adopts the running ORACLE stage (its own process group, .pid_OR/.rc_OR; same 15-min stall watchdog, resume),
# then (a) DFDC per the procedure fixed before re-running (PROGRESS 01/11 17:0x): verify on 300 seed-0 random
# frames with the unchanged criterion -> batched if met, else every frame with the author's single-image function;
# then starts the web (it stays stopped between ORACLE and (a)).
set -u
cd "$(dirname "$0")/../.."
export R=${R:-/home/aiotlab/hoangan/eval_runs/root_fix/e2}
export PROGRESS=${PROGRESS:-/home/aiotlab/hoangan/eval_runs/root_fix/PROGRESS.md}
R6_SUPERVISOR_LIB=1 source tools/v3/r6_supervisor.sh
REPO=$(pwd)
note() { echo "- $(date '+%d/%m %H:%M') E2 watchdog: $*" >> "$PROGRESS"; log "$*"; }
eval "$(sed -n '/^web_start() {/,/^}/p' tools/v3/e2_supervisor.sh)"
WEBLOG=/tmp/safevision_final.log
echo $$ > "$R/supervisor.pid"
J=$R/jobs
OR_CMD="$PY -u tools/v3/r6_batched.py --frames $J/OR_changed.csv --system V3 --lora $L --splits test --out $R/results_OR.csv >> $R/results_OR.log 2>&1"

# adopt the running ORACLE stage
pg=$(cat "$R/.pid_OR"); start=$(date +%s)
log "adopt OR (process group $pg)"
while [ ! -f "$R/.rc_OR" ]; do
    sleep "$POLL"; [ -f "$R/.rc_OR" ] && break
    now=$(date +%s); last=$(newest "$R/results_OR.csv" "$R/results_OR.log"); [ "$last" -lt "$start" ] && last=$start
    if [ $((now - last)) -gt "$STALL_SEC" ]; then
        note "OR không tiến trong $(( (now - last) / 60 )) phút → kill nhóm $pg của mình, chạy lại từ chỗ dở"
        kill -TERM -- -"$pg" 2>/dev/null; sleep 31; kill -KILL -- -"$pg" 2>/dev/null; echo stalled > "$R/.rc_OR"
    fi
done
if [ "$(cat "$R/.rc_OR")" = "stalled" ]; then run_stage OR "$R/results_OR.csv $R/results_OR.log" "$OR_CMD"
elif [ "$(cat "$R/.rc_OR")" = "0" ]; then log "done OR"
else note "OR THẤT BẠI (mã $(cat "$R/.rc_OR"))"; FAILED="$FAILED OR"; fi

A="$PY -u tools/v3/r6_batched.py --frames $J/a_new.csv --system a --questions $R/a_questions.csv --splits test --out $R/results_a.csv"
run_stage a_verify "$R/results_a.log" "$A --verify 300 --verify-random --verify-only >> $R/results_a.log 2>&1"
if "$PY" -c "import json,sys; d=json.load(open('$R/results_a.meta.json')); sys.exit(int(not(d['mean_abs_diff']<=1e-3 and d['p99_abs_diff']<=5e-3 and d['verdict_agree']==d['n_verify'])))"; then
    note "(a) verify 300 ngẫu nhiên ĐẠT ($("$PY" -c "import json; d=json.load(open('$R/results_a.meta.json')); print(f\"TB {d['mean_abs_diff']:.2e}, p99 {d['p99_abs_diff']:.2e}, verdict {d['verdict_agree']}/{d['n_verify']}\")")) ⇒ chấm theo lô"
    run_stage a "$R/results_a.csv $R/results_a.log" "$A >> $R/results_a.log 2>&1"
else
    note "(a) verify 300 ngẫu nhiên KHÔNG đạt ($("$PY" -c "import json; d=json.load(open('$R/results_a.meta.json')); print(f\"TB {d['mean_abs_diff']:.2e}, p99 {d['p99_abs_diff']:.2e}, verdict {d['verdict_agree']}/{d['n_verify']}\")")) ⇒ chấm từng ảnh (hàm tác giả) toàn bộ DFDC"
    run_stage a "$R/results_a.csv $R/results_a.log" "$A --single >> $R/results_a.log 2>&1"
fi
web_start
if [ -n "$FAILED" ]; then note "chuỗi E2 (phần 2) kết thúc, stage thất bại:$FAILED"; exit 1; fi
note "chuỗi E2 xong (P0, P1, ORACLE, a); web đã chạy lại."
