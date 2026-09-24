#!/usr/bin/env bash
set -u

while tmux has-session -t "router4_paper_2cross_cdf_dfdcp" 2>/dev/null
do
    if ! "/home/aiotlab/miniconda3/envs/X2DFD/bin/python" "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/guard_2cross_runtime.py" "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP" "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/STOP_REASON.txt" >> "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/guard_2cross.log" 2>&1
    then
        echo "DFDC guard triggered" >> "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/guard_2cross.log"
        tmux kill-session -t "router4_paper_2cross_cdf_dfdcp" 2>/dev/null || true
        exit 1
    fi

    sleep 30
done
