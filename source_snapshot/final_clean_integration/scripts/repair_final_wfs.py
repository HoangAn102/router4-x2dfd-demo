from pathlib import Path
import pandas as pd
import subprocess
import os
import re
import time
import shutil
import sys
import json

BASE=Path("/home/aiotlab/hoangan")
ROOT=BASE/"outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

TRAIN=ROOT/"01_teacher/train_teacher_FINAL_CLEAN.csv"
ROUTER=ROOT/"02_router/final_checkpoint/best.pt"

WFSDIR=ROOT/"03_wfs"
WFS=WFSDIR/"router4_FINAL_CLEAN_routed_wfs.csv"

RX=BASE/"projects/router_x2dfd"
PYTHON="/home/aiotlab/miniconda3/envs/X2DFD/bin/python"

WFSDIR.mkdir(parents=True,exist_ok=True)

print("="*80)
print("LOAD CLEAN TRAIN")
print("="*80)

base=pd.read_csv(TRAIN,low_memory=False)

if "image_path" not in base.columns:
    raise RuntimeError("clean teacher has no image_path")

N=len(base)

print("CLEAN TRAIN ROWS =",N)

if N >= 259084:
    print(
        "[INFO] clean train row count is unexpectedly close "
        "to old raw train; continuing with exact validation."
    )


# ============================================================
# A. ROUTER LIVE EXPORT
# ============================================================

print()
print("="*80)
print("A. ROUTER LIVE EXPORT ON CLEAN TRAIN")
print("="*80)

candidates=[
    RX/"route_test_locked038.py",
    RX/"eval_router4_pilot_val.py",
    RX/"eval_router_full_val.py",
]

pred_path=None

for source in candidates:

    if not source.is_file():
        continue

    print()
    print("TRY:",source)

    src=source.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Route scripts generally read one manifest directly.
    # Replace DIRECT literal pd.read_csv paths only.
    src=re.sub(
        r'''pd\.read_csv\(\s*(?:r|R)?["'][^"']+\.csv["']''',
        'pd.read_csv(r"'+str(TRAIN)+'"',
        src
    )

    # Replace known teacher CSV literals.
    src=re.sub(
        r'''["']/home/aiotlab/hoangan/outputs/router_teacher_v2/(?:train|val)_teacher_v2\.csv["']''',
        'r"'+str(TRAIN)+'"',
        src
    )

    # Replace Router checkpoint literals, never .pth expert weights.
    def replace_pt(m):
        quote=m.group(1)
        path=m.group(2)

        low=path.lower()

        if path.endswith(".pt") and (
            "router" in low
            or path.endswith("/best.pt")
        ):
            return quote+str(ROUTER)+quote

        return m.group(0)

    src=re.sub(
        r'''(["'])([^"']+\.pt)\1''',
        replace_pt,
        src
    )

    patched=WFSDIR/f"patched_{source.name}"
    patched.write_text(src,encoding="utf-8")

    started=time.time()

    log=WFSDIR/f"{source.stem}_router_export.log"

    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]="0"

    with open(log,"w") as f:
        proc=subprocess.run(
            [PYTHON,str(patched)],
            cwd=str(RX),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print("EXIT =",proc.returncode)

    if proc.returncode != 0:
        print("FAILED; see",log)
        continue

    # Find newly written CSVs only.
    found=[]

    search_roots=[
        ROOT,
        BASE/"outputs",
        RX,
    ]

    seen=set()

    for root in search_roots:

        if not root.exists():
            continue

        for cur,dirs,files in os.walk(root):

            try:
                depth=len(
                    Path(cur).relative_to(root).parts
                )
            except Exception:
                depth=0

            if depth>5:
                dirs[:]=[]
                continue

            dirs[:]=[
                d for d in dirs
                if d not in {
                    ".git","__pycache__",
                    "datasets","dataset",
                    "weights","images"
                }
                and not d.startswith("checkpoint-")
            ]

            for fn in files:

                if not fn.endswith(".csv"):
                    continue

                p=Path(cur)/fn

                sp=str(p)

                if sp in seen:
                    continue

                seen.add(sp)

                try:
                    st=p.stat()

                    if st.st_mtime < started-2:
                        continue

                    head=pd.read_csv(
                        p,
                        nrows=2,
                        low_memory=False
                    )

                except Exception:
                    continue

                cols=set(head.columns)

                if (
                    "image_path" in cols
                    and (
                        "router_expert_idx" in cols
                        or "selected_expert" in cols
                    )
                ):
                    found.append(p)


    if not found:
        print("No fresh router prediction CSV from this candidate.")
        continue


    # Validate candidates by row count.
    for p in sorted(
        found,
        key=lambda x:x.stat().st_mtime,
        reverse=True
    ):

        try:
            x=pd.read_csv(
                p,
                usecols=lambda c:
                    c in {
                        "image_path",
                        "router_expert_idx",
                        "selected_expert",
                        "router_confidence",
                        "router_margin",
                    },
                low_memory=False
            )
        except Exception:
            continue

        print("candidate:",p,"rows=",len(x))

        if len(x)==N:
            pred_path=p
            break

    if pred_path is not None:
        break


if pred_path is None:
    print()
    print("[FAIL] Could not export clean Router predictions.")

    for source in candidates:
        if source.is_file():
            print("Candidate source:",source)

    raise SystemExit(20)


print()
print("ROUTER PRED =",pred_path)


# ============================================================
# B. ALIGN ROUTER PREDICTIONS WITH CLEAN TEACHER
# ============================================================

print()
print("="*80)
print("B. ALIGN CLEAN ROUTER PREDICTIONS")
print("="*80)

pred=pd.read_csv(
    pred_path,
    low_memory=False
)

if len(pred)!=N:
    raise RuntimeError(
        f"Router rows {len(pred)} != clean rows {N}"
    )

experts=[
    "blending",
    "diffusion",
    "frequency",
    "texture",
]


if "router_expert_idx" not in pred.columns:

    if "selected_expert" not in pred.columns:
        raise RuntimeError(
            "router output has neither router_expert_idx "
            "nor selected_expert"
        )

    mapping={
        e:i
        for i,e in enumerate(experts)
    }

    pred["router_expert_idx"]=(
        pred["selected_expert"]
        .astype(str)
        .str.lower()
        .map(mapping)
    )


if pred["router_expert_idx"].isna().any():
    raise RuntimeError(
        "some Router predictions have invalid expert index"
    )


# Prefer exact row order.
same_order=(
    pred["image_path"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        base["image_path"]
        .astype(str)
        .reset_index(drop=True)
    )
)


if same_order:

    merged=base.copy()

    for c in pred.columns:

        if c.startswith("router_") or c=="selected_expert":
            merged[c]=pred[c].values

else:

    # Fail rather than produce a many-to-many merge.
    if (
        base["image_path"].duplicated().any()
        or pred["image_path"].duplicated().any()
    ):
        raise RuntimeError(
            "image order differs and duplicate paths prevent "
            "safe key merge"
        )

    keep=[
        c for c in pred.columns
        if (
            c=="image_path"
            or c.startswith("router_")
            or c=="selected_expert"
        )
    ]

    merged=base.merge(
        pred[keep],
        on="image_path",
        how="left",
        validate="1:1"
    )

    if merged["router_expert_idx"].isna().any():
        raise RuntimeError(
            "Router predictions do not cover clean teacher exactly"
        )


merged_path=WFSDIR/"clean_teacher_plus_NEW_router.csv"

merged.to_csv(
    merged_path,
    index=False
)

print("MERGED =",merged_path)
print(
    "ROUTER DISTRIBUTION:",
    merged["router_expert_idx"]
    .value_counts()
    .sort_index()
    .to_dict()
)


# ============================================================
# C. REBUILD WFS USING THE PROJECT'S REAL WFS BUILDER
# ============================================================

print()
print("="*80)
print("C. REBUILD WFS")
print("="*80)

builder=RX/"build_router4_train_wfs.py"

if not builder.is_file():
    raise FileNotFoundError(builder)

src=builder.read_text(
    encoding="utf-8",
    errors="ignore"
)

# Replace direct CSV input with merged clean/new-router manifest.
src=re.sub(
    r'''pd\.read_csv\(\s*(?:r|R)?["'][^"']+\.csv["']''',
    'pd.read_csv(r"'+str(merged_path)+'"',
    src
)

# Known old manifest paths.
src=re.sub(
    r'''["']/home/aiotlab/hoangan/outputs/[^"']+\.csv["']''',
    'r"'+str(merged_path)+'"',
    src
)

# Restore OUTPUT path specifically.
old_output=(
    "/home/aiotlab/hoangan/outputs/"
    "x2dfd_router4_training/wfs/train_routed_wfs.csv"
)

src=src.replace(
    'r"'+str(merged_path)+'"',
    'r"'+str(merged_path)+'"',
)

# Replace output filename/path wherever literal name occurs.
src=src.replace(
    old_output,
    str(WFS)
)

# General old output directory.
src=src.replace(
    "/home/aiotlab/hoangan/outputs/x2dfd_router4_training/wfs",
    str(WFSDIR)
)

patched_builder=WFSDIR/"build_router4_train_wfs_FINAL_FIXED.py"

patched_builder.write_text(
    src,
    encoding="utf-8"
)

started=time.time()

log=WFSDIR/"wfs_FINAL_FIXED.log"

with open(log,"w") as f:

    proc=subprocess.run(
        [PYTHON,str(patched_builder)],
        cwd=str(RX),
        stdout=f,
        stderr=subprocess.STDOUT,
    )

print("BUILDER EXIT =",proc.returncode)

if proc.returncode != 0:
    print(Path(log).read_text(errors="ignore")[-5000:])
    raise SystemExit(30)


# Builder may still have another output variable.
# Find only FRESH WFS CSVs.
fresh=[]

for root in [
    WFSDIR,
    BASE/"outputs",
]:

    for cur,dirs,files in os.walk(root):

        try:
            depth=len(
                Path(cur).relative_to(root).parts
            )
        except Exception:
            depth=0

        if depth>5:
            dirs[:]=[]
            continue

        dirs[:]=[
            d for d in dirs
            if d not in {
                ".git","datasets","dataset",
                "weights","images"
            }
        ]

        for fn in files:

            if not fn.endswith(".csv"):
                continue

            p=Path(cur)/fn

            try:
                st=p.stat()

                if st.st_mtime < started-2:
                    continue

                h=pd.read_csv(
                    p,
                    nrows=1,
                    low_memory=False
                )

            except Exception:
                continue

            if (
                "wfs_text" in h.columns
                and "selected_expert" in h.columns
            ):
                fresh.append(p)


if WFS.is_file():
    candidate=WFS

elif fresh:

    candidate=max(
        fresh,
        key=lambda x:x.stat().st_mtime
    )

    shutil.copy2(
        candidate,
        WFS
    )

else:

    print(
        Path(log).read_text(
            errors="ignore"
        )[-5000:]
    )

    raise RuntimeError(
        "Builder ran but no fresh WFS CSV was produced"
    )


# ============================================================
# D. HARD VALIDATION
# ============================================================

print()
print("="*80)
print("D. FINAL WFS VALIDATION")
print("="*80)

w=pd.read_csv(
    WFS,
    low_memory=False
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

missing=required-set(w.columns)

if missing:
    raise RuntimeError(
        "WFS missing columns: "+repr(missing)
    )

if len(w)!=N:
    raise RuntimeError(
        f"FINAL WFS rows={len(w)}, CLEAN TRAIN={N}"
    )


if not (
    w["image_path"].astype(str).reset_index(drop=True)
    .equals(
        base["image_path"].astype(str).reset_index(drop=True)
    )
):
    raise RuntimeError(
        "FINAL WFS image_path sequence != clean train"
    )


# Must actually carry NEW Router selections.
if not (
    w["selected_expert"]
    .astype(str)
    .str.lower()
    .isin(experts)
    .all()
):
    raise RuntimeError(
        "invalid selected_expert values"
    )


(WFSDIR/"ROUTED_WFS_PATH.txt").write_text(
    str(WFS)+"\n"
)

(WFSDIR/"WFS_READY").touch()


report={
    "clean_train_rows":N,
    "router_prediction_source":str(pred_path),
    "final_wfs":str(WFS),
    "final_wfs_rows":len(w),
    "selected_expert_distribution":
        w["selected_expert"]
        .value_counts()
        .to_dict(),
    "status":"PASS",
}

(WFSDIR/"FINAL_WFS_AUDIT.json").write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
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
        ensure_ascii=False
    )
)

