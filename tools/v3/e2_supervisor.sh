#!/bin/bash
# E2 supervisor (PREREG_E2): same watchdog/run_stage as r6_supervisor.sh (15 min no progress -> kill own process
# group, note in PROGRESS, resume; error exit -> stage failed, not retried; done stages skipped).
# Order: wait for the DFDC raw-score and author-question jobs (by PID) -> Router V3 probs -> job files ->
# stop the web (own uvicorn on :8010 only, times in PROGRESS) -> LLaVA stages P0, P1, a, ORACLE -> start the web.
#   E2_WAIT_PIDS="pid1 pid2" setsid nohup tools/v3/e2_supervisor.sh > $R/supervisor.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
REPO=$(pwd)
export R=${R:-/home/aiotlab/hoangan/eval_runs/root_fix/e2}
export PROGRESS=${PROGRESS:-/home/aiotlab/hoangan/eval_runs/root_fix/PROGRESS.md}
R6_SUPERVISOR_LIB=1 source tools/v3/r6_supervisor.sh
note() { echo "- $(date '+%d/%m %H:%M') E2 watchdog: $*" >> "$PROGRESS"; log "$*"; }
WEBLOG=/tmp/safevision_final.log

web_stop() {
    local pid pgid
    pid=$(ss -ltnp 2>/dev/null | grep '127.0.0.1:8010 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -z "$pid" ] && { note "web không chạy (8010 trống) — không cần dừng"; return 0; }
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [ "$(ps -o user= -p "$pid" | tr -d ' ')" = "$(whoami)" ] || { note "8010 không phải tiến trình của mình — KHÔNG dừng, bỏ E2 LLaVA"; return 1; }
    kill -TERM -- -"$pgid"; for _ in $(seq 1 30); do ps -p "$pid" > /dev/null || break; sleep 1; done
    note "dừng web (uvicorn PID $pid, nhóm $pgid) để chạy LLaVA E2"
}
web_start() {
    (cd "$REPO" && setsid nohup "$PY" -c 'import os; os.environ["RUN_MODE"]="live"; os.environ["FACE_CROP_BACKEND"]="dlib"; import uvicorn; uvicorn.run("web.backend.app.main:app", host="127.0.0.1", port=8010, workers=1)' >> "$WEBLOG" 2>&1 < /dev/null &)
    sleep 20
    note "chạy lại web $(git -C "$REPO" log --oneline -1 | cut -c1-7): $(ss -ltnp 2>/dev/null | grep '127.0.0.1:8010 ' | grep -o 'pid=[0-9]*' | head -1); health $(curl -s http://127.0.0.1:8010/api/v1/health | head -c 60)"
}

echo $$ > "$R/supervisor.pid"
for p in ${E2_WAIT_PIDS:-}; do
    log "wait for PID $p"; while kill -0 "$p" 2>/dev/null; do sleep 30; done
done
grep -q "score_blending" "$R/frames_e2.csv" && "$PY" -c "import pandas as pd,sys; d=pd.read_csv('$R/frames_e2.csv'); sys.exit(int(d[[c for c in d if c.startswith('score_')]].isna().any().any()))" \
    || { note "DFDC raw scores chưa đủ — dừng E2"; exit 1; }
run_stage route "$R/frames_e2.csv $R/route.log" "$PY -u tools/v3/e2_frames.py route --out $R >> $R/route.log 2>&1"
run_stage jobs "$R/jobs.log" "$PY -u tools/v3/e2_frames.py jobs --out $R >> $R/jobs.log 2>&1"
[ -n "$FAILED" ] && { note "E2 dừng trước LLaVA, thất bại:$FAILED"; exit 1; }
"$PY" -c "import pandas as pd,sys; q=pd.read_csv('$R/a_questions.csv'); j=pd.read_csv('$R/jobs/a_new.csv'); sys.exit(int(not set(j.image_path)<=set(q.image_path)))" \
    || { note "câu hỏi tác giả DFDC chưa đủ — dừng E2"; exit 1; }
web_stop || exit 1
J=$R/jobs
run_stage P0 "$R/results_P0.csv $R/results_P0.log" \
    "$PY -u tools/v3/r6_batched.py --frames $J/P0_new.csv --system V3 --lora $L --splits test --out $R/results_P0.csv >> $R/results_P0.log 2>&1"
run_stage P1 "$R/results_P1.csv $R/results_P1.log" \
    "$PY -u tools/v3/r6_batched.py --frames $J/P1_changed.csv --system V3 --lora $L --splits test --out $R/results_P1.csv >> $R/results_P1.log 2>&1"
run_stage a "$R/results_a.csv $R/results_a.log" \
    "$PY -u tools/v3/r6_batched.py --frames $J/a_new.csv --system a --questions $R/a_questions.csv --splits test --out $R/results_a.csv >> $R/results_a.log 2>&1"
run_stage OR "$R/results_OR.csv $R/results_OR.log" \
    "$PY -u tools/v3/r6_batched.py --frames $J/OR_changed.csv --system V3 --lora $L --splits test --out $R/results_OR.csv >> $R/results_OR.log 2>&1"
web_start
if [ -n "$FAILED" ]; then note "chuỗi E2 kết thúc, stage thất bại:$FAILED"; exit 1; fi
note "chuỗi E2 xong (route, jobs, P0, P1, a, ORACLE); web đã chạy lại."
