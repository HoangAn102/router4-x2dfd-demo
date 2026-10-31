#!/bin/bash
# R6 (full) supervisor: runs the stages one after another (one LLaVA job at a time, ~16 GB VRAM).
# Watchdog: a stage whose watched files (output + log) have not changed for STALL_SEC (default 15 min) is
# killed (its own process group only), the stall is written to PROGRESS.md and the stage is restarted; every
# stage resumes from what it already wrote (max TRIES stalls). A stage that EXITS with an error is not retried
# (a config/code error repeats; it must be fixed first): it is marked failed in PROGRESS.md, the next
# (independent) stages still run, and the supervisor exits 1 at the end. Stages already done (.rc_NAME = 0)
# are skipped, so re-running the supervisor after a fix only redoes what is missing.
# Stages are tracked by PID / exit-code files, never by pgrep on command lines.
#   setsid nohup tools/v3/r6_supervisor.sh > $R/supervisor.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
R=${R:-/home/aiotlab/hoangan/eval_runs/root_fix/r6_full}
PROGRESS=${PROGRESS:-/home/aiotlab/hoangan/eval_runs/root_fix/PROGRESS.md}
PY=${PY:-/home/aiotlab/miniconda3/envs/X2DFD/bin/python}
L=${L:-/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030/06_lora/router4_x2dfd_V3_full30k}
STALL_SEC=${STALL_SEC:-900}
POLL=${POLL:-60}
TRIES=${TRIES:-5}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "$(date '+%d/%m %H:%M:%S') $*"; }
note() { echo "- $(date '+%d/%m %H:%M') R6 watchdog: $*" >> "$PROGRESS"; log "$*"; }
newest() { local m=0 t; for f in "$@"; do t=$(stat -c %Y "$f" 2>/dev/null || echo 0); [ "$t" -gt "$m" ] && m=$t; done; echo $m; }

# run_stage NAME "WATCHED FILES" CMD...
run_stage() {
    local name=$1 watch=$2; shift 2
    local rc_file="$R/.rc_$name" try pgid start rc
    if [ "$(cat "$rc_file" 2>/dev/null)" = "0" ]; then log "skip $name (already done)"; return 0; fi
    for try in $(seq 1 "$TRIES"); do
        rm -f "$rc_file"
        log "start $name (try $try)"
        setsid bash -c "( $* ); echo \$? > '$rc_file'" < /dev/null > /dev/null 2>&1 &
        pgid=$!
        echo "$pgid" > "$R/.pid_$name"
        start=$(date +%s)
        while [ ! -f "$rc_file" ]; do
            sleep "$POLL"
            [ -f "$rc_file" ] && break
            local now last; now=$(date +%s); last=$(newest $watch)
            [ "$last" -lt "$start" ] && last=$start
            if [ $((now - last)) -gt "$STALL_SEC" ]; then
                note "$name không tiến trong $(( (now - last) / 60 )) phút (lần $try) → kill nhóm tiến trình $pgid của mình, chạy lại từ chỗ dở. Log cuối: $(tail -c 300 ${watch##* } 2>/dev/null | tr '\r\n' '  ' | tail -c 200)"
                kill -TERM -- -"$pgid" 2>/dev/null; sleep $((POLL / 2 + 1)); kill -KILL -- -"$pgid" 2>/dev/null
                echo stalled > "$rc_file"
            fi
        done
        rc=$(cat "$rc_file")
        if [ "$rc" = "0" ]; then log "done $name"; return 0; fi
        if [ "$rc" != "stalled" ]; then
            note "$name THẤT BẠI (thoát mã $rc, không phải treo) → không chạy lại; sửa nguyên nhân rồi chạy lại supervisor (phần đã xong được bỏ qua/resume). Lỗi: $(grep -E 'Error|error|Traceback' ${watch##* } 2>/dev/null | tail -2 | tr '\n' ' ' | tail -c 250)"
            FAILED="$FAILED $name"
            return 1
        fi
        sleep $((POLL / 2 + 1))
    done
    note "$name treo $TRIES lần — bỏ stage, cần xem log."
    FAILED="$FAILED $name"
    return 1
}
FAILED=""

[ -n "${R6_SUPERVISOR_LIB:-}" ] && return 0  # sourced by the test: functions only
echo $$ > "$R/supervisor.pid"
F="$R/frames_full.csv"
run_stage V3 "$R/results_V3.csv $R/results_V3.log" \
    "$PY -u tools/v3/r6_batched.py --frames $F --system V3 --lora $L --out $R/results_V3.csv >> $R/results_V3.log 2>&1"
run_stage a "$R/results_a.csv $R/results_a.log" \
    "$PY -u tools/v3/r6_batched.py --frames $F --system a --questions $R/a_questions.csv --out $R/results_a.csv >> $R/results_a.log 2>&1"
run_stage bold "$R/results_bold.csv $R/results_bold.log" \
    "$PY -u tools/v3/r6_batched.py --frames $F --system bold --splits test --out $R/results_bold.csv >> $R/results_bold.log 2>&1"
run_stage V3_gray "$R/results_V3_gray.log" \
    "$PY -u tools/v3/r6_llava.py --frames $F --system V3_gray --lora-v3 $L --out $R/results_V3_gray.csv >> $R/results_V3_gray.log 2>&1"
run_stage tf_existing "$R/tf_check/tf_existing_full.log" \
    "$PY -u tools/v3/check_tf_equivalence.py --part existing --n 0 --src $R/results_V3.pre_crash_59788.csv --tag _all59788 --out $R/tf_check >> $R/tf_check/tf_existing_full.log 2>&1"
if [ -n "$FAILED" ]; then note "chuỗi R6 kết thúc, stage thất bại:$FAILED"; exit 1; fi
note "chuỗi R6 xong (V3, a, bold, V3_gray, kiểm TF 59.788 dòng)."
