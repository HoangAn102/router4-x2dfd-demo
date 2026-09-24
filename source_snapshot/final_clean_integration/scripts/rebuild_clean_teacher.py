from pathlib import Path
from collections import Counter
import json, re, math
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


BASE=Path("/home/aiotlab/hoangan")
ROOT=BASE/"outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919"

MASTER=BASE/"outputs/ROUTER4_MASTER_CLEAN_20260919"
TRAIN=MASTER/"MASTER_CLEAN_TRAIN.csv"
VAL=MASTER/"MASTER_CLEAN_VAL.csv"

OLD_TEACHER=BASE/"outputs/router_teacher_v2"
OLD_CAL=OLD_TEACHER/"calibrators_train_only.joblib"
OLD_TRAIN=OLD_TEACHER/"train_teacher_v2.csv"

CLEAN=ROOT/"00_clean"
TOUT=ROOT/"01_teacher"

CLEAN.mkdir(parents=True,exist_ok=True)
TOUT.mkdir(parents=True,exist_ok=True)

EXPERTS=["blending","diffusion","frequency","texture"]


def norm(x):
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except: pass
    s=str(x).strip()
    return "" if s.lower() in {"","nan","none","null","<na>"} else s


def source_key(r):
    p=norm(r.get("image_path")).replace("\\","/")
    if not p: return ""

    name=Path(p).name
    domain=norm(r.get("domain")).upper()
    gen=norm(r.get("generator")).lower()

    # FF++: video / sequence parent
    if domain=="FFPP" or "FF++" in domain:
        return "FFPP:"+str(Path(p).parent)

    if gen=="attgan" or "/AttGAN/" in p:
        m=re.match(r"^(\d+)_\d+\.[^.]+$",name)
        if m:
            return "ATTGAN:"+m.group(1)

    if gen=="stgan" or "/STGAN/" in p:
        m=re.match(r"^(\d+)_",name)
        if m:
            return "STGAN:"+m.group(1)

    return ""


print("LOAD MASTER CLEAN")
tr=pd.read_csv(TRAIN,low_memory=False)
va=pd.read_csv(VAL,low_memory=False)

print("MASTER TRAIN",len(tr))
print("MASTER VAL  ",len(va))

###############################################################################
# MAXIMUM-CONSERVATIVE VALIDATION
#
# Training keeps all 3 project domains.
# Internal validation keeps only authoritative FF++ source/video cohort.
#
# Unknown-lineage StyleGAN/GANGen rows are NOT used for model selection.
# They can still be training data and later be evaluated on independent TEST.
###############################################################################

domain = va["domain"].astype(str).str.upper()

strict_val = va[
    domain.eq("FFPP") | domain.str.contains("FF\\+\\+",regex=True)
].copy()

if len(strict_val)==0:
    raise RuntimeError(
        "Could not identify FFPP validation rows. Refusing to invent split."
    )

strict_train=tr.copy()

train_keys=set()
val_keys=set()

for _,r in strict_train.iterrows():
    k=source_key(r)
    if k:
        train_keys.add(k)

for _,r in strict_val.iterrows():
    k=source_key(r)
    if k:
        val_keys.add(k)

overlap=train_keys & val_keys

if overlap:
    bad=[]
    for i,r in strict_train.iterrows():
        if source_key(r) in overlap:
            bad.append(i)

    strict_train=strict_train.drop(index=bad).reset_index(drop=True)

    # Re-check
    train_keys=set(
        source_key(r)
        for _,r in strict_train.iterrows()
        if source_key(r)
    )

    overlap=train_keys & val_keys

if overlap:
    raise RuntimeError(
        f"Known source overlap remains: {len(overlap)}"
    )


STRICT_TRAIN=CLEAN/"MAX_CLEAN_TRAIN.csv"
STRICT_VAL=CLEAN/"MAX_CLEAN_VAL.csv"

strict_train.to_csv(STRICT_TRAIN,index=False)
strict_val.to_csv(STRICT_VAL,index=False)

gate={
    "master_train":len(tr),
    "max_clean_train":len(strict_train),
    "master_val":len(va),
    "strict_internal_val":len(strict_val),
    "known_source_overlap":len(overlap),
    "train_domains":strict_train["domain"].value_counts(dropna=False).to_dict(),
    "val_domains":strict_val["domain"].value_counts(dropna=False).to_dict(),
    "protocol":
        "Train uses all cleaned project domains; internal validation is "
        "restricted to authoritative FF++ source/video-disjoint cohort. "
        "Unknown-lineage cohorts are not used for model selection."
}

json.dump(
    gate,
    open(CLEAN/"MAX_CLEAN_GATE.json","w"),
    indent=2,
    ensure_ascii=False
)

print(json.dumps(gate,indent=2,ensure_ascii=False))


###############################################################################
# REBUILD CALIBRATION + TEACHER FROM CLEAN DATA
###############################################################################

if "label" not in strict_train.columns:
    raise RuntimeError("label column missing")

y=strict_train["label"].astype(int).to_numpy()
yv=strict_val["label"].astype(int).to_numpy()

for e in EXPERTS:
    c=f"score_{e}"
    if c not in strict_train.columns:
        raise RuntimeError(
            f"Missing raw expert score column: {c}"
        )


old_cal=None

if OLD_CAL.is_file():
    try:
        old_cal=joblib.load(OLD_CAL)
        print("Loaded previous calibrator artifact:",type(old_cal))
    except Exception as exc:
        print("Could not load previous calibrator:",repr(exc))


def find_estimator(obj, expert):
    if obj is None:
        return None

    if hasattr(obj,"fit") and (
        hasattr(obj,"predict") or hasattr(obj,"predict_proba")
    ):
        return obj

    if isinstance(obj,dict):
        # direct key first
        for k,v in obj.items():
            if expert in str(k).lower():
                z=find_estimator(v,expert)
                if z is not None:
                    return z

        # then recursive
        for v in obj.values():
            z=find_estimator(v,expert)
            if z is not None:
                return z

    if isinstance(obj,(list,tuple)):
        for v in obj:
            z=find_estimator(v,expert)
            if z is not None:
                return z

    return None


def fit_model(template,x,y):
    x=np.asarray(x,dtype=np.float64).reshape(-1,1)

    if template is not None:
        try:
            est=clone(template)
            est.fit(x,y)
            return est,"CLONED_"+type(template).__name__
        except Exception:
            pass

    # Safe generic monotonicity-independent fallback.
    est=LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=20260919
    )
    est.fit(x,y)
    return est,"LogisticRegression_FALLBACK"


def pred_fake(est,x):
    x=np.asarray(x,dtype=np.float64).reshape(-1,1)

    if hasattr(est,"predict_proba"):
        p=np.asarray(est.predict_proba(x))

        if p.ndim==2 and p.shape[1]>=2:
            return p[:,1]

    p=np.asarray(est.predict(x),dtype=np.float64).reshape(-1)
    return np.clip(p,1e-6,1-1e-6)


skf=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=20260919
)

final_models={}
calibration_info={}

for e in EXPERTS:
    col=f"score_{e}"

    x=strict_train[col].astype(float).to_numpy()
    xv=strict_val[col].astype(float).to_numpy()

    template=find_estimator(old_cal,e)

    oof=np.zeros(len(strict_train),dtype=np.float64)

    methods=[]

    for a,b in skf.split(x,y):
        est,method=fit_model(
            template,
            x[a],
            y[a]
        )

        methods.append(method)

        oof[b]=pred_fake(
            est,
            x[b]
        )

    final,method=fit_model(
        template,
        x,
        y
    )

    final_models[e]=final

    strict_train[f"cal_{e}"]=np.clip(
        oof,
        1e-5,
        1-1e-5
    )

    strict_val[f"cal_{e}"]=np.clip(
        pred_fake(final,xv),
        1e-5,
        1-1e-5
    )

    calibration_info[e]={
        "template":
            None if template is None
            else type(template).__name__,

        "methods":
            sorted(set(methods+[method]))
    }


###############################################################################
# UTILITY = calibrated probability assigned to true class
###############################################################################

for df in [strict_train,strict_val]:
    labels=df["label"].astype(int).to_numpy()

    for e in EXPERTS:
        p=df[f"cal_{e}"].astype(float).to_numpy()

        u=np.where(
            labels==1,
            p,
            1-p
        )

        df[f"utility_{e}"]=np.clip(
            u,
            1e-5,
            1-1e-5
        )


###############################################################################
# Recover teacher softmax temperature from OLD teacher CSV
###############################################################################

T=0.10
temperature_source="fallback_0.10"

try:
    old=pd.read_csv(
        OLD_TRAIN,
        nrows=25000,
        low_memory=False
    )

    soft_cols=[
        f"teacher_soft{i}"
        for i in range(4)
    ]

    util_cols=[
        f"utility_{e}"
        for e in EXPERTS
    ]

    if all(c in old.columns for c in soft_cols+util_cols):

        U=old[util_cols].astype(float).to_numpy()
        S=old[soft_cols].astype(float).to_numpy()

        good=np.isfinite(U).all(1)&np.isfinite(S).all(1)

        U=U[good]
        S=S[good]

        best=(1e99,None)

        for t in np.logspace(
            np.log10(.01),
            np.log10(2.0),
            160
        ):
            Z=U/t
            Z=Z-Z.max(1,keepdims=True)
            P=np.exp(Z)
            P=P/P.sum(1,keepdims=True)

            mse=float(
                np.mean(
                    (P-S)**2
                )
            )

            if mse<best[0]:
                best=(mse,float(t))

        if best[1] is not None:
            T=best[1]
            temperature_source="recovered_from_old_teacher"

except Exception as exc:
    print("Temperature recovery failed:",repr(exc))


print("Teacher temperature:",T,temperature_source)


def build_teacher(df):
    U=np.column_stack(
        [
            df[f"utility_{e}"].astype(float).to_numpy()
            for e in EXPERTS
        ]
    )

    hard=U.argmax(1)

    Z=U/T
    Z=Z-Z.max(1,keepdims=True)

    soft=np.exp(Z)
    soft=soft/soft.sum(1,keepdims=True)

    df=df.copy()

    df["teacher_hard_idx"]=hard
    df["teacher_hard"]=[
        EXPERTS[i]
        for i in hard
    ]

    df["teacher_best_expert"]=df["teacher_hard"]
    df["teacher_best_utility"]=U.max(1)

    for i,e in enumerate(EXPERTS):
        df[f"teacher_soft{i}"]=soft[:,i]
        df[f"teacher_soft_{e}"]=soft[:,i]

    return df


strict_train=build_teacher(strict_train)
strict_val=build_teacher(strict_val)


TRAIN_OUT=TOUT/"train_teacher_FINAL_CLEAN.csv"
VAL_OUT=TOUT/"val_teacher_FINAL_CLEAN.csv"

strict_train.to_csv(
    TRAIN_OUT,
    index=False
)

strict_val.to_csv(
    VAL_OUT,
    index=False
)

joblib.dump(
    {
        "experts":final_models,
        "calibration_info":calibration_info,
        "teacher_temperature":T,
        "temperature_source":temperature_source,
        "protocol":"5-fold OOF clean calibration; final models fitted on all clean train"
    },
    TOUT/"calibrators_FINAL_CLEAN.joblib"
)

summary={
    "train_rows":len(strict_train),
    "val_rows":len(strict_val),
    "teacher_temperature":T,
    "temperature_source":temperature_source,
    "calibration":calibration_info,
    "hard_distribution":
        strict_train["teacher_hard"].value_counts().to_dict()
}

json.dump(
    summary,
    open(TOUT/"TEACHER_FINAL_SUMMARY.json","w"),
    indent=2,
    ensure_ascii=False
)

print("="*80)
print("FINAL CLEAN TEACHER READY")
print("="*80)
print(TRAIN_OUT)
print(VAL_OUT)
print(json.dumps(summary,indent=2,ensure_ascii=False))
