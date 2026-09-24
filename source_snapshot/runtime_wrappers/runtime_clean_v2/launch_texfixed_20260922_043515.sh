#!/usr/bin/env bash
set -o pipefail

unset LD_LIBRARY_PATH
unset PYTHONPATH
unset PYTHONHOME

export X2DFD_PEFT_COMPAT_ROOT="/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/peft_compat"
export USE_PROGRESS_BAR=1

echo "============================================================"
echo "ROUTER4 × X2DFD CLEAN V2 — TEXTURE FIXED"
date
echo "driver=/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/RUN_2CROSS_CLEAN_REBUILD_V2.sh"
echo "ambient_LD_LIBRARY_PATH=${LD_LIBRARY_PATH-<unset>}"
echo "============================================================"

"/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/RUN_2CROSS_CLEAN_REBUILD_V2.sh" 2>&1 | tee -a "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/logs/CLEAN_V2_TEXFIXED_20260922_043515.log"

RC=${PIPESTATUS[0]}

echo
echo "============================================================"
echo "DRIVER_RC=$RC"
date
echo "============================================================"

nvidia-smi || true

exec /bin/bash --noprofile --norc
