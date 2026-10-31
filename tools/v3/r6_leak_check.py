"""R6 PREREG §6: leakage between the R6 TEST frames and every training/fitting set of the compared lineages.

For each TEST set x training source: exact path overlap, video-id overlap, identity overlap (FFPP source/target ids,
Celeb-DF idX, DFF person id = leading number of the IMDB-WIKI file name) and sha256 content overlap.

    python tools/v3/r6_leak_check.py --frames r6_full/frames_full.csv --out r6_full/leak_check
"""
import argparse
import hashlib
import json
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

O = Path("/home/aiotlab/hoangan/outputs")
V3, D, FC = O / "ROUTER4_V3_20261030", O / "ROUTER4_DIFF_RETRAIN_20261030", O / "ROUTER4_FINAL_CLEAN_RETRAIN_20260919"
SOURCES = {  # name -> (file, what it trained/fitted)
    "V3_router_teacher_calib_train": (V3 / "02_teacher/train_teacher_V3.csv", "Router V3 train + calibrator V3 fit"),
    "V3_calib_scores_train": (V3 / "01_calibrators/scores_V3_train.csv", "calibrator V3 fit"),
    "V3_router_val": (V3 / "02_teacher/val_teacher_V3.csv", "Router V3 model selection (DEV-side)"),
    "V3_SFT": (V3 / "05_sft/router4_x2dfd_V3_SFT.json", "LoRA V3 SFT (30,000)"),
    "interim_router_train": (D / "01_teacher/train_teacher_v3.csv", "Router v3 (interim) train"),
    "interim_diffusion_oof": (D / "01_teacher/diffusion_oof.csv", "calibrator D1 diffusion refit"),
    "b_router_teacher_train": (O / "router_teacher_v2/train_teacher_v2.csv", "Router v2 train + calibrator v2 fit"),
    "b_max_clean_train": (FC / "00_clean/MAX_CLEAN_TRAIN.csv", "FINAL_CLEAN clean TRAIN (calibrator FINAL_CLEAN)"),
    "b_SFT_FINAL_CLEAN": (FC / "04_sft/router4_x2dfd_FINAL_CLEAN_SFT.json", "LoRA FINAL_CLEAN SFT (24,000)"),
}


def load_paths(f):
    if f.suffix == ".json":
        return [it["image"] for it in json.load(open(f))]
    return pd.read_csv(f, usecols=["image_path"]).image_path.tolist()


def video_id(p):
    p = str(p)
    if "FFPP_X2DFD32_CROPPED" in p or "/frames/" in p:
        return Path(p).parent.name if Path(p).suffix else Path(p).name
    return Path(p).stem


def identities(p):
    p, v = str(p), video_id(p)
    if "FaceForensics" in p:
        return {f"ffpp:{x}" for x in re.findall(r"(?<![0-9])([0-9]{3})(?![0-9])", v.split("__")[-1])}
    if "Celeb-DF" in p:
        return {f"cdf:{x}" for x in re.findall(r"id[0-9]+", v)}
    if "DeepFakeFace" in p:
        m = re.match(r"([0-9]+)_", Path(p).name)
        return {f"dff:{m.group(1)}"} if m else set()
    return set()


def sha(p):
    try:
        return p, hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return p, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(a.frames, usecols=["split", "dataset", "video", "image_path"])
    test = fr[fr.split == "test"]
    src = {k: sorted(set(load_paths(f))) for k, (f, _) in SOURCES.items() if f.exists()}
    missing = [k for k, (f, _) in SOURCES.items() if not f.exists()]

    cache = a.out / "sha256_cache.csv"
    hashes = dict(pd.read_csv(cache).values) if cache.exists() else {}
    todo = sorted(set(test.image_path) | set().union(*src.values()))
    todo = [p for p in todo if p not in hashes]
    with Pool(a.procs) as pool:
        for i, (p, h) in enumerate(pool.imap_unordered(sha, todo, chunksize=256)):
            hashes[p] = h
            if i % 50000 == 0:
                print("hashed", i, "/", len(todo), flush=True)
    pd.DataFrame(sorted(hashes.items()), columns=["path", "sha256"]).to_csv(cache, index=False)

    rep, rows = {"missing_sources": missing, "sources": {k: dict(file=str(SOURCES[k][0]), role=SOURCES[k][1],
                                                                     n=len(v)) for k, v in src.items()}}, []
    for k, paths in src.items():
        s_path, s_vid = set(paths), {video_id(p) for p in paths}
        s_id = set().union(*map(identities, paths))
        s_hash = {hashes[p] for p in paths if hashes.get(p)}
        s_unread = sum(hashes.get(p) is None for p in paths)
        for ds, g in test.groupby("dataset"):
            g_ids = [identities(p) for p in g.image_path]
            hit_hash = g.image_path.map(lambda p: hashes.get(p) in s_hash)
            r = dict(test=ds, source=k, frames=len(g), videos=g.video.nunique(),
                     path_overlap=int(g.image_path.isin(s_path).sum()),
                     video_overlap=int(g.image_path.map(video_id).isin(s_vid).sum()),
                     identity_overlap_frames=int(sum(bool(x & s_id) for x in g_ids)),
                     identity_overlap_ids=sorted(set().union(*g_ids) & s_id)[:20],
                     sha256_overlap_frames=int(hit_hash.sum()),
                     sha256_overlap_videos=int(g.video[hit_hash].nunique()), source_unreadable=int(s_unread))
            rows.append(r)
            print({kk: vv for kk, vv in r.items() if kk != "identity_overlap_ids"}, flush=True)
    rep["test_unreadable"] = int(sum(hashes.get(p) is None for p in test.image_path))
    rep["pairs"] = rows
    # frames to exclude: any path/video/sha256 overlap with any source (identity is reported, see REPORT)
    excl = set()
    for k, paths in src.items():
        s_hash = {hashes[p] for p in paths if hashes.get(p)}
        s_path = set(paths)
        excl |= set(test.image_path[test.image_path.isin(s_path) | test.image_path.map(lambda p: hashes.get(p) in s_hash)])
    rep["exclude_frames"] = len(excl)
    pd.Series(sorted(excl), name="image_path").to_csv(a.out / "exclude_frames.csv", index=False)
    (a.out / "leak_report.json").write_text(json.dumps(rep, indent=1))
    pd.DataFrame(rows).drop(columns=["identity_overlap_ids"]).to_csv(a.out / "leak_pairs.csv", index=False)


if __name__ == "__main__":
    main()
