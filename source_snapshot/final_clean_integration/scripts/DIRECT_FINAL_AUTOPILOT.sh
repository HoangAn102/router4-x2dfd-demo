#!/usr/bin/env bash
set -euo pipefail

BASE=/home/aiotlab/hoangan
ENV=/home/aiotlab/miniconda3/envs/X2DFD
PY="$ENV/bin/python"

ROOT="$BASE/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

TRAIN="$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv"
NEW_ROUTER="$ROOT/02_router/final_checkpoint/best.pt"
OLD_ROUTER="$BASE/outputs/router_training_v2/run_effb0_3domain/best.pt"

RX="$BASE/projects/router_x2dfd"

WFSDIR="$ROOT/03_wfs"

ROUTED_INPUT="$WFSDIR/clean_train_NEW_router.csv"
FINAL_WFS="$WFSDIR/router4_FINAL_CLEAN_routed_wfs.csv"

EXPORTER="$ROOT/scripts/direct_full_router_export.py"

ORIGINAL_RUNNER="$ROOT/scripts/run_full_final.sh"

log () {
    echo
    echo "======================================================================"
    echo "$1"
    echo "$(date)"
    echo "======================================================================"
}

fail () {
    echo
    echo "[STOP] $1"

    printf '%s\n' "$1" \
      > "$ROOT/DIRECT_FINAL_STOP_REASON.txt"

    exit 1
}


###############################################################################
# 1. VERIFY LOCKED CLEAN ARTIFACTS
###############################################################################

log "1. VERIFY CLEAN TEACHER / CALIBRATOR / ROUTER"

test -s "$TRAIN" \
    || fail "Clean teacher TRAIN missing"

test -s "$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib" \
    || fail "Final calibrator missing"

test -s "$NEW_ROUTER" \
    || fail "Final clean Router missing"

test -s "$OLD_ROUTER" \
    || fail "Old Router missing for parity verification"

echo "Clean teacher:"
ls -lh "$TRAIN"

echo
echo "Calibrator:"
ls -lh "$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib"

echo
echo "New Router:"
ls -lh "$NEW_ROUTER"


###############################################################################
# 2. DIRECT FULL ROUTER EXPORT
###############################################################################

log "2. DIRECT ROUTER EXPORT — FULL CLEAN TRAIN"

rm -f \
  "$WFSDIR/WFS_READY" \
  "$WFSDIR/ROUTED_WFS_PATH.txt" \
  "$FINAL_WFS"

export CUDA_VISIBLE_DEVICES=0

"$PY" "$EXPORTER" \
    --train "$TRAIN" \
    --old-router "$OLD_ROUTER" \
    --new-router "$NEW_ROUTER" \
    --out "$ROUTED_INPUT" \
    --batch-size 256 \
    --workers 12 \
    >> "$ROOT/logs/direct_router_export.log" 2>&1

test -s "$ROUTED_INPUT" \
    || fail "Direct Router output missing"

echo "DIRECT ROUTER OUTPUT:"
ls -lh "$ROUTED_INPUT"


###############################################################################
# 3. BUILD WFS USING REAL PROJECT BUILDER
###############################################################################

log "3. BUILD FINAL CLEAN WFS"

BUILDER="$RX/build_router4_train_wfs.py"
PATCHED="$WFSDIR/build_router4_train_wfs_DIRECT_FINAL.py"

test -s "$BUILDER" \
    || fail "build_router4_train_wfs.py missing"


"$PY" - \
  "$BUILDER" \
  "$PATCHED" \
  "$ROUTED_INPUT" \
  "$FINAL_WFS" <<'PY'
from pathlib import Path
import sys,re

src_path=Path(sys.argv[1])
dst=Path(sys.argv[2])

inp=sys.argv[3]
out=sys.argv[4]

s=src_path.read_text(
    encoding="utf-8",
    errors="ignore",
)

# Replace known historical train input paths.
known_inputs=[
    "/home/aiotlab/hoangan/outputs/router_teacher_v2/train_teacher_v2.csv",
    "/home/aiotlab/hoangan/outputs/x2dfd_router4_training/train24k_manifest.csv",
]

for p in known_inputs:
    s=s.replace(
        p,
        inp,
    )

# Replace direct pd.read_csv literal(s) only when they clearly
# reference TRAIN/router manifest.
def read_repl(m):
    full=m.group(0)
    path=m.group(1)

    low=path.lower()

    if (
        "train" in low
        or "teacher" in low
        or "manifest" in low
        or "router4" in low
    ):
        return (
            'pd.read_csv(r"'
            + inp
            + '"'
        )

    return full

s=re.sub(
    r'''pd\.read_csv\(\s*(?:r|R)?["']([^"']+\.csv)["']''',
    read_repl,
    s,
)

# Redirect historical WFS output.
s=s.replace(
    "/home/aiotlab/hoangan/outputs/x2dfd_router4_training/wfs/train_routed_wfs.csv",
    out,
)

s=s.replace(
    "/home/aiotlab/hoangan/outputs/x2dfd_router4_training/wfs",
    str(Path(out).parent),
)

dst.write_text(
    s,
    encoding="utf-8",
)

print("PATCHED:",dst)
PY


"$PY" "$PATCHED" \
    >> "$ROOT/logs/direct_wfs_build.log" 2>&1


###############################################################################
# 4. FIND + HARD-VALIDATE EXACT FINAL WFS
###############################################################################

log "4. HARD VALIDATE FINAL WFS"

"$PY" - \
  "$TRAIN" \
  "$ROUTED_INPUT" \
  "$FINAL_WFS" \
  "$WFSDIR" <<'PY'
from pathlib import Path
import pandas as pd
import sys,os,shutil,json,time

train=Path(sys.argv[1])
routed=Path(sys.argv[2])
final=Path(sys.argv[3])
folder=Path(sys.argv[4])

a=pd.read_csv(
    train,
    usecols=["image_path"],
    low_memory=False,
)

N=len(a)

# If hardcoded builder wrote somewhere else, find only fresh-ish
# candidate with exact row count + required columns.
candidates=[]

if final.is_file():
    candidates.append(final)

roots=[
    folder,
    Path("/home/aiotlab/hoangan/outputs/x2dfd_router4_training/wfs"),
]

for root in roots:
    if not root.exists():
        continue

    for p in root.glob("*.csv"):
        if p in candidates:
            continue

        try:
            h=pd.read_csv(
                p,
                nrows=1,
                low_memory=False,
            )

            cols=set(h.columns)

            if {
                "image_path",
                "selected_expert",
                "wfs_text",
            }.issubset(cols):
                candidates.append(p)

        except Exception:
            pass


chosen=None

for p in sorted(
    candidates,
    key=lambda x:x.stat().st_mtime,
    reverse=True,
):

    try:
        b=pd.read_csv(
            p,
            usecols=[
                "image_path",
                "selected_expert",
                "wfs_text",
            ],
            low_memory=False,
        )
    except Exception:
        continue

    print(
        "WFS CANDIDATE:",
        p,
        "rows=",
        len(b),
    )

    if len(b) != N:
        continue

    if not (
        b["image_path"]
        .astype(str)
        .reset_index(drop=True)
        .equals(
            a["image_path"]
            .astype(str)
            .reset_index(drop=True)
        )
    ):
        continue

    if b["wfs_text"].isna().any():
        continue

    chosen=p
    break


if chosen is None:
    raise RuntimeError(
        "No WFS candidate exactly matches clean TRAIN"
    )


if chosen.resolve() != final.resolve():
    shutil.copy2(
        chosen,
        final,
    )


# Full validation of final file.
full=pd.read_csv(
    final,
    low_memory=False,
)

required={
    "image_path",
    "selected_expert",
    "wfs_text",
    "cal_blending",
    "cal_diffusion",
    "cal_frequency",
    "cal_texture",
}

missing=required-set(full.columns)

if missing:
    raise RuntimeError(
        "Missing WFS columns: "
        + repr(missing)
    )

if len(full)!=N:
    raise RuntimeError(
        f"WFS rows={len(full)}, expected={N}"
    )

if not (
    full["image_path"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        a["image_path"]
        .astype(str)
        .reset_index(drop=True)
    )
):
    raise RuntimeError(
        "WFS image paths/order differ from clean TRAIN"
    )

allowed={
    "blending",
    "diffusion",
    "frequency",
    "texture",
}

if not (
    full["selected_expert"]
    .astype(str)
    .str.lower()
    .isin(allowed)
    .all()
):
    raise RuntimeError(
        "Invalid selected_expert values"
    )


(folder/"ROUTED_WFS_PATH.txt").write_text(
    str(final)+"\n"
)

(folder/"WFS_READY").touch()

report={
    "status":"PASS",
    "rows":len(full),
    "wfs":str(final),
    "distribution":
        full["selected_expert"]
        .value_counts()
        .to_dict(),
}

(folder/"DIRECT_FINAL_WFS_AUDIT.json").write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("="*80)
print("✅ FINAL CLEAN WFS READY")
print("="*80)
print(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
)
PY


###############################################################################
# 5. REBUILD SFT + TRAIN FINAL LORA
###############################################################################

log "5. REBUILD FINAL SFT + TRAIN FINAL LORA"

test -s "$ORIGINAL_RUNNER" \
    || fail "Original full runner missing"

DLINE=$(
    grep -n '^# D\. REBUILD CLEAN SFT' \
      "$ORIGINAL_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

FLINE=$(
    grep -n '^# F\. SAME-LIST COMPARISON' \
      "$ORIGINAL_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

ALINE=$(
    grep -n '^# A\. CLEAN' \
      "$ORIGINAL_RUNNER" \
    | head -1 \
    | cut -d: -f1
)

[ -n "$DLINE" ] \
    || fail "Stage D marker missing"

[ -n "$FLINE" ] \
    || fail "Stage F marker missing"

[ -n "$ALINE" ] \
    || fail "Stage A marker missing"


CONTINUE="$ROOT/scripts/DIRECT_SFT_LORA_FINAL.sh"

head -n $((ALINE-1)) \
    "$ORIGINAL_RUNNER" \
    > "$CONTINUE"


cat >> "$CONTINUE" <<EOF

###############################################################################
# DIRECT FINAL OVERRIDES
###############################################################################

export CUDA_VISIBLE_DEVICES=0

ROUTER_BEST="$NEW_ROUTER"
ROUTED="$FINAL_WFS"

echo "======================================================================"
echo "DIRECT FINAL: SFT -> LoRA"
echo "======================================================================"

echo "Router: \$ROUTER_BEST"
echo "WFS   : \$ROUTED"
EOF


sed -n \
  "${DLINE},$((FLINE-1))p" \
  "$ORIGINAL_RUNNER" \
  >> "$CONTINUE"


###############################################################################
# FREEZE
###############################################################################

cat >> "$CONTINUE" <<'EOF'

log "6. FREEZE FINAL MODEL"

FINAL_MODEL="$ROOT/05_lora/router4_x2dfd_FINAL_CLEAN"
FREEZE="$ROOT/05_lora/DEPLOY_FINAL_CLEAN"

test -s "$FINAL_MODEL/adapter_model.safetensors" \
    || fail "Final adapter missing"

test -s "$FINAL_MODEL/adapter_config.json" \
    || fail "Final adapter config missing"


rm -rf "$FREEZE"
mkdir -p "$FREEZE"


for f in \
    adapter_model.safetensors \
    adapter_config.json \
    non_lora_trainables.bin \
    config.json \
    trainer_state.json \
    README.md \
    MODEL_READY
do
    if [ -e "$FINAL_MODEL/$f" ]; then
        cp -a \
          "$FINAL_MODEL/$f" \
          "$FREEZE/"
    fi
done


sha256sum \
  "$FINAL_MODEL/adapter_model.safetensors" \
  > "$FREEZE/adapter_model.sha256"


touch "$FREEZE/FINAL_MODEL_READY"


PKG="$ROOT/05_lora/ROUTER4_FINAL_CLEAN_MODEL.tar.gz"

rm -f "$PKG"

tar -C "$ROOT/05_lora" \
    -czf "$PKG" \
    DEPLOY_FINAL_CLEAN


sha256sum "$PKG" \
    > "$PKG.sha256"


cat > "$ROOT/FINAL_MODEL_STATUS.txt" <<STATUS
============================================================
ROUTER4 X2DFD FINAL CLEAN
============================================================

Clean teacher:
$ROOT/01_teacher/train_teacher_FINAL_CLEAN.csv

Clean calibrator:
$ROOT/01_teacher/calibrators_FINAL_CLEAN.joblib

Final Router:
$ROUTER_BEST

Final clean WFS:
$ROUTED

Final SFT:
$(cat "$ROOT/04_sft/FINAL_SFT_PATH.txt" 2>/dev/null || true)

Final LoRA:
$FINAL_MODEL

Deploy:
$FREEZE

Package:
$PKG

STATUS:
FINAL MODEL READY
STATUS


touch "$ROOT/FINAL_MODEL_READY"


echo
echo "======================================================================"
echo "✅✅✅ FINAL MODEL READY ✅✅✅"
echo "======================================================================"

cat "$ROOT/FINAL_MODEL_STATUS.txt"

EOF


chmod +x "$CONTINUE"
bash -n "$CONTINUE" \
    || fail "SFT/LoRA continuation syntax invalid"


bash "$CONTINUE" \
    >> "$ROOT/logs/direct_sft_lora.log" 2>&1


###############################################################################
# DONE
###############################################################################

log "DIRECT FINAL AUTOPILOT COMPLETE"

test -f "$ROOT/FINAL_MODEL_READY" \
    || fail "Pipeline ended without FINAL_MODEL_READY"

cat "$ROOT/FINAL_MODEL_STATUS.txt"

