"""In-process evaluation of the documented web pipeline, comparing two face-crop modes.

    upload -> face crop -> Router4 -> selected expert -> calibrator -> X2DFD (verdict = fake_prob >= 0.5)

Same classes as the web (FaceCropper, RouterClient, ExpertClient, X2DFDClient); explanations are off and
experts run batched per expert (like analyze_video_batch). Face-crop modes:
  haar   strict Haar only (pipeline before the YuNet fallback)
  yunet  Haar, then YuNet when Haar finds nothing or a tiny box (what the web runs now)
LLaVA is loaded once (stop the web first) and scores each distinct crop once, shared across modes.

Usage:
    python tools/eval/pipeline_eval.py --manifest M.csv --out DIR            # stages 1-3, no LLaVA
    python tools/eval/pipeline_eval.py --manifest M.csv --out DIR --x2dfd    # adds the LLaVA stage
Manifest columns: path, group, label (0 real / 1 fake), role.
"""
import argparse
import asyncio
import csv
import hashlib
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PIL import Image, ImageOps  # noqa: E402

from web.backend.app.services.face_cropper import FaceCropper  # noqa: E402
from web.model_worker.expert_client import ExpertClient  # noqa: E402
from web.model_worker.router_client import RouterClient  # noqa: E402
from web.model_worker.x2dfd_client import X2DFDClient  # noqa: E402

MODES = {"haar": False, "yunet": True}   # name -> FaceCropper fallback flag
FIELDS = ["path", "group", "label", "role", "face", "detector", "crop", "crop_sha", "expert", "alias",
          "router_conf", "margin", "expert_raw", "expert_cal", "x2dfd_fake_prob", "verdict"]


def isnum(v):
    try:
        return v not in ("", None) and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def read_rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stage_face_router(rows, crops_dir, fallback, router):
    for i, r in enumerate(rows):
        if r.get("face") not in ("", None) and (r["face"] == "0" or isnum(r.get("router_conf"))):
            continue
        img = ImageOps.exif_transpose(Image.open(r["path"])).convert("RGB")
        found = FaceCropper.crop_image_with_detector(img, fallback=fallback)
        if found is None:
            r.update(face="0", detector="", crop="", crop_sha="")
            continue
        crop, detector = found
        cp = (crops_dir / f"{i:05d}.png").resolve()
        crop.save(cp)
        expert, alias, conf, margin = router.route_image(cp)
        r.update(face="1", detector=detector, crop=str(cp), crop_sha=hashlib.sha1(crop.tobytes()).hexdigest(),
                 expert=expert, alias=alias, router_conf=conf, margin=margin)


async def stage_experts(rows):
    ec = ExpertClient()
    todo = [r for r in rows if r.get("crop") and not isnum(r.get("expert_raw"))]
    for e in ("blending", "diffusion", "frequency", "texture"):
        rs = [r for r in todo if r["expert"] == e]
        if rs:
            res = await ec.compute_expert_scores_batch(e, [Path(r["crop"]) for r in rs])
            for r in rs:
                r["expert_raw"], r["expert_cal"] = res[str(Path(r["crop"]).resolve())]


async def stage_x2dfd(all_rows):
    """One LLaVA session; a crop (same pixels, expert, calibrated score) is scored once for all modes."""
    client = X2DFDClient()
    done = {}
    try:
        for r in all_rows:
            if isnum(r.get("x2dfd_fake_prob")) and r.get("crop"):
                done[(r["crop_sha"], r["expert"], round(float(r["expert_cal"]), 6))] = float(r["x2dfd_fake_prob"])
        todo = [r for r in all_rows if r.get("crop") and not isnum(r.get("x2dfd_fake_prob"))]
        t0 = time.time()
        for n, r in enumerate(todo):
            key = (r["crop_sha"], r["expert"], round(float(r["expert_cal"]), 6))
            if key not in done:
                res = await client.infer(Path(r["crop"]), r["alias"], float(r["expert_cal"]), include_explanation=False)
                done[key] = res.fake_probability
            r["x2dfd_fake_prob"] = done[key]
            if (n + 1) % 100 == 0:
                print(f"  x2dfd {n + 1}/{len(todo)} rows, {len(done)} distinct crops, {time.time() - t0:.0f}s", flush=True)
    finally:
        await client.stop_worker()


def stage_verdict(rows):
    for r in rows:
        if not r.get("crop"):
            r["verdict"] = "NO_FACE"
        elif isnum(r.get("x2dfd_fake_prob")):
            r["verdict"] = "FAKE" if float(r["x2dfd_fake_prob"]) >= 0.5 else "REAL"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--x2dfd", action="store_true", help="run the LLaVA stage (stop the web first)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    router = RouterClient()
    per_mode = {}
    t0 = time.time()
    for mode, fallback in MODES.items():
        crops = args.out / f"crops_{mode}"
        crops.mkdir(exist_ok=True)
        prior = {r["path"]: r for r in read_rows(args.out / f"results_{mode}.csv")}
        rows = []
        for m in read_rows(args.manifest):
            r = dict(prior.get(m["path"], {}))
            r.update({k: m.get(k, "") for k in ("path", "group", "label", "role")})
            rows.append(r)
        stage_face_router(rows, crops, fallback, router)
        asyncio.run(stage_experts(rows))
        per_mode[mode] = rows
        print(f"[{mode}] faces {sum(r['face'] == '1' for r in rows)}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    if args.x2dfd:
        asyncio.run(stage_x2dfd([r for rows in per_mode.values() for r in rows]))
    for mode, rows in per_mode.items():
        stage_verdict(rows)
        write_rows(args.out / f"results_{mode}.csv", rows)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
