from pathlib import Path
import runpy
import pandas as pd
import shutil
import time
import os
import json

BASE=Path("/home/aiotlab/hoangan")
ROOT=BASE/"outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

BUILDER=BASE/"projects/router_x2dfd/build_router4_train_wfs.py"
INPUT=ROOT/"03_wfs/clean_train_NEW_router.csv"
OUTPUT=ROOT/"03_wfs/router4_FINAL_CLEAN_routed_wfs.csv"
TRAIN=ROOT/"01_teacher/train_teacher_FINAL_CLEAN.csv"

if not INPUT.is_file():
    raise FileNotFoundError(INPUT)

orig_read_csv = pd.read_csv
orig_to_csv = pd.DataFrame.to_csv

print("="*80)
print("FORCED FINAL WFS BUILD")
print("="*80)
print("INPUT :",INPUT)
print("OUTPUT:",OUTPUT)

# ------------------------------------------------------------
# Redirect historical TRAIN manifest reads only.
# ------------------------------------------------------------

def forced_read_csv(filepath_or_buffer, *args, **kwargs):

    if isinstance(filepath_or_buffer,(str,Path)):

        s=str(filepath_or_buffer)
        low=s.lower()

        try:
            same=Path(s).resolve()==INPUT.resolve()
        except Exception:
            same=False

        if not same and s.lower().endswith(".csv"):

            historical=(
                "x2dfd_router4_training" in low
                or "router_teacher_v2/train_teacher_v2" in low
                or "train24k_manifest" in low
                or "router4_train_manifest" in low
                or "train_routed_wfs" in low
            )

            if historical:
                print(
                    "[READ REDIRECT]",
                    s,
                    "->",
                    INPUT,
                    flush=True,
                )

                return orig_read_csv(
                    INPUT,
                    *args,
                    **kwargs
                )

    return orig_read_csv(
        filepath_or_buffer,
        *args,
        **kwargs
    )


# ------------------------------------------------------------
# Force routed/WFS CSV output to canonical FINAL_WFS.
# ------------------------------------------------------------

def forced_to_csv(self, path_or_buf=None, *args, **kwargs):

    if isinstance(path_or_buf,(str,Path)):

        s=str(path_or_buf)
        low=s.lower()

        if (
            "wfs" in low
            or "routed" in low
        ):
            print(
                "[WRITE REDIRECT]",
                s,
                "->",
                OUTPUT,
                flush=True,
            )

            OUTPUT.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            return orig_to_csv(
                self,
                OUTPUT,
                *args,
                **kwargs
            )

    return orig_to_csv(
        self,
        path_or_buf,
        *args,
        **kwargs
    )


pd.read_csv = forced_read_csv
pd.DataFrame.to_csv = forced_to_csv

start=time.time()

runpy.run_path(
    str(BUILDER),
    run_name="__main__",
)

# Restore pandas just for validation.
pd.read_csv = orig_read_csv
pd.DataFrame.to_csv = orig_to_csv


# ------------------------------------------------------------
# If builder wrote another fresh file despite monkeypatch,
# find only an EXACT 254121-row WFS candidate.
# ------------------------------------------------------------

base=orig_read_csv(
    TRAIN,
    usecols=["image_path"],
    low_memory=False,
)

N=len(base)

def valid_candidate(p):
    try:
        h=orig_read_csv(
            p,
            nrows=1,
            low_memory=False,
        )

        required={
            "image_path",
            "selected_expert",
            "wfs_text",
        }

        if not required.issubset(h.columns):
            return False

        x=orig_read_csv(
            p,
            usecols=[
                "image_path",
                "selected_expert",
                "wfs_text",
            ],
            low_memory=False,
        )

        if len(x)!=N:
            return False

        if not (
            x["image_path"]
            .astype(str)
            .reset_index(drop=True)
            .equals(
                base["image_path"]
                .astype(str)
                .reset_index(drop=True)
            )
        ):
            return False

        return True

    except Exception:
        return False


if not OUTPUT.is_file() or not valid_candidate(OUTPUT):

    candidates=[]

    for root in [
        ROOT/"03_wfs",
        BASE/"outputs/x2dfd_router4_training/wfs",
    ]:
        if not root.exists():
            continue

        for p in root.glob("*.csv"):

            try:
                if p.stat().st_mtime < start-2:
                    continue
            except Exception:
                continue

            if valid_candidate(p):
                candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "Builder completed but produced no WFS "
            f"matching clean TRAIN ({N} rows)."
        )

    chosen=max(
        candidates,
        key=lambda p:p.stat().st_mtime,
    )

    shutil.copy2(
        chosen,
        OUTPUT,
    )


# ------------------------------------------------------------
# HARD FINAL VALIDATION
# ------------------------------------------------------------

full=orig_read_csv(
    OUTPUT,
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
        "FINAL WFS missing columns: "
        + repr(missing)
    )

if len(full)!=N:
    raise RuntimeError(
        f"FINAL WFS rows={len(full)}, expected={N}"
    )

if not (
    full["image_path"]
    .astype(str)
    .reset_index(drop=True)
    .equals(
        base["image_path"]
        .astype(str)
        .reset_index(drop=True)
    )
):
    raise RuntimeError(
        "FINAL WFS image sequence does not match clean TRAIN"
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
        "FINAL WFS contains invalid selected_expert"
    )

if full["wfs_text"].isna().any():
    raise RuntimeError(
        "FINAL WFS contains empty wfs_text"
    )

audit={
    "status":"PASS",
    "rows":len(full),
    "output":str(OUTPUT),
    "distribution":
        full["selected_expert"]
        .value_counts()
        .to_dict(),
}

(ROOT/"03_wfs/DIRECT_FINAL_WFS_AUDIT.json").write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)

(ROOT/"03_wfs/ROUTED_WFS_PATH.txt").write_text(
    str(OUTPUT)+"\n"
)

(ROOT/"03_wfs/WFS_READY").touch()

print()
print("="*80)
print("✅ FINAL CLEAN WFS READY")
print("="*80)
print(json.dumps(audit,indent=2,ensure_ascii=False))
