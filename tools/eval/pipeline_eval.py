"""In-process evaluation of the web pipeline, using the same classes and the same decide() as the web.

Stages (each cached in --out, rerun resumes):
  1 face    FaceCropper.crop_image_with_detector  (Haar -> YuNet -> dlib, as the image endpoint)
  2 ai      AIImageDetector on the full image
  3 router  RouterClient (selected expert, confidence, margin)
  4 experts selected expert and blending, batched per expert (like analyze_video_batch)
  5 x2dfd   X2DFDClient.infer(include_explanation=False)   [--x2dfd; loads LLaVA, stop the web first]
  6 decide  services.decision.decide() -> final verdict

Usage:
    python tools/eval/pipeline_eval.py --manifest M.csv --out DIR            # stages 1-4 (+6 gate-only columns)
    python tools/eval/pipeline_eval.py --manifest M.csv --out DIR --x2dfd    # adds stage 5 and the final verdict
Manifest columns: path, group, label (0 real / 1 fake), role.
"""
import argparse
import asyncio
import csv
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image, ImageOps  # noqa: E402

from web.backend.app.schemas.image import FinalResult, Reliability  # noqa: E402
from web.backend.app.services.decision import decide  # noqa: E402
from web.backend.app.services.face_cropper import FaceCropper  # noqa: E402
from web.backend.app.services.reliability import assess_reliability  # noqa: E402
from web.model_worker.ai_image_detector import AIImageDetector  # noqa: E402
from web.model_worker.expert_client import ExpertClient  # noqa: E402
from web.model_worker.router_client import RouterClient  # noqa: E402
from web.model_worker.x2dfd_client import X2DFDClient  # noqa: E402

EXPERTS = ["blending", "diffusion", "frequency", "texture"]
FIELDS = [
    "path", "group", "label", "role", "face", "detector", "crop", "ai_score", "expert", "alias", "router_conf",
    "margin", "expert_raw", "expert_cal", "blending_raw", "x2dfd_fake_prob", "verdict", "x2dfd_verdict",
    "gate", "reliability", "t_x2dfd_s",
]
NAN = float("nan")


def read_rows(path: Path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def write_rows(path: Path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def isnum(v):
    try:
        return v not in ("", None) and not math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def stage_face_ai_router(rows, crops_dir):
    ai = AIImageDetector()
    router = RouterClient()
    for i, r in enumerate(rows):
        if r.get("face") not in ("", None) and isnum(r.get("router_conf")) and isnum(r.get("ai_score")):
            continue
        img = ImageOps.exif_transpose(Image.open(r["path"])).convert("RGB")
        found = FaceCropper.crop_image_with_detector(img)
        r["ai_score"] = ai.score(Path(r["path"]))
        if found is None:
            r.update(face="0", detector="", crop="")
            continue
        crop, detector = found
        cp = (crops_dir / f"{i:05d}.png").resolve()
        crop.save(cp)
        expert, alias, conf, margin = router.route_image(cp)
        r.update(face="1", detector=detector, crop=str(cp), expert=expert, alias=alias, router_conf=conf, margin=margin)
        if (i + 1) % 100 == 0:
            print(f"  face/ai/router {i + 1}/{len(rows)}", flush=True)


async def stage_experts(rows):
    ec = ExpertClient()
    todo = [r for r in rows if r.get("crop") and not isnum(r.get("expert_raw"))]
    if not todo:
        return
    # selected expert (raw + calibrated) for X2DFD's prompt
    for e in EXPERTS:
        rs = [r for r in todo if r["expert"] == e]
        if not rs:
            continue
        t0 = time.time()
        res = await ec.compute_expert_scores_batch(e, [Path(r["crop"]) for r in rs])
        for r in rs:
            r["expert_raw"], r["expert_cal"] = res[str(Path(r["crop"]).resolve())]
        print(f"  expert {e}: {len(rs)} images in {time.time() - t0:.0f}s", flush=True)
    # blending raw on every crop (reused when blending was selected)
    rest = [r for r in todo if r["expert"] != "blending"]
    for r in todo:
        if r["expert"] == "blending":
            r["blending_raw"] = r["expert_raw"]
    if rest:
        t0 = time.time()
        res = await ec.compute_expert_scores_batch("blending", [Path(r["crop"]) for r in rest])
        for r in rest:
            r["blending_raw"] = res[str(Path(r["crop"]).resolve())][0]
        print(f"  blending (extra): {len(rest)} images in {time.time() - t0:.0f}s", flush=True)


async def stage_x2dfd(rows, only_groups, out_csv):
    client = X2DFDClient()
    try:
        todo = [r for r in rows if r.get("crop") and not isnum(r.get("x2dfd_fake_prob"))
                and (only_groups is None or r["group"] in only_groups)]
        t_start = time.time()
        for n, r in enumerate(todo):
            t0 = time.time()
            res = await client.infer(Path(r["crop"]), r["alias"], float(r["expert_cal"]), include_explanation=False)
            r["x2dfd_fake_prob"], r["x2dfd_verdict"], r["t_x2dfd_s"] = res.fake_probability, res.verdict, round(time.time() - t0, 2)
            if (n + 1) % 20 == 0:
                write_rows(out_csv, rows)
                print(f"  x2dfd {n + 1}/{len(todo)} ({(time.time() - t_start) / (n + 1):.1f}s/img)", flush=True)
    finally:
        await client.stop_worker()


def stage_decide(rows):
    for r in rows:
        if not r.get("crop"):
            # No face: only the AI gate can judge (same rule as analyze_faceless_image).
            from web.backend.app.services.ai_gate import build_faceless_response
            fake = build_faceless_response("eval", float(r["ai_score"]), {}, {})
            r["verdict"], r["gate"] = ("FAKE", "ai") if fake else ("NO_FACE", "")
            continue
        bl = float(r["blending_raw"]) if isnum(r.get("blending_raw")) else None
        ai = float(r["ai_score"]) if isnum(r.get("ai_score")) else None
        if not isnum(r.get("x2dfd_fake_prob")):
            # Gates only (no LLaVA): what the gates alone would flag.
            f = FinalResult(verdict="REAL", fake_probability=0.0, real_probability=1.0, continuous_score=0.0)
            final, _, ai_d, bl_d = decide(f, Reliability(), ai, bl)
            r["verdict"] = "GATES_FAKE" if final.verdict == "FAKE" else "GATES_PASS"
            r["gate"] = "ai" if ai_d and ai_d.overridden else ("blending" if bl_d and bl_d.overridden else "")
            continue
        p = float(r["x2dfd_fake_prob"])
        f = FinalResult(verdict="FAKE" if p >= 0.5 else "REAL", fake_probability=p, real_probability=1.0 - p, continuous_score=p)
        final, rel, ai_d, bl_d = decide(f, assess_reliability(float(r["router_conf"]), float(r["expert_cal"])), ai, bl)
        r["verdict"], r["reliability"] = final.verdict, rel.level
        r["gate"] = "ai" if ai_d and ai_d.overridden else ("blending" if bl_d and bl_d.overridden else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--x2dfd", action="store_true", help="run the LLaVA stage (stop the web first)")
    ap.add_argument("--x2dfd-groups", nargs="*", default=None, help="restrict the LLaVA stage to these groups")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    crops = args.out / "crops"
    crops.mkdir(exist_ok=True)
    out_csv = args.out / "results.csv"

    prior = {r["path"]: r for r in read_rows(out_csv)}
    rows = []
    for m in read_rows(args.manifest):
        r = dict(prior.get(m["path"], {}))
        r.update({k: m.get(k, "") for k in ("path", "group", "label", "role")})
        rows.append(r)

    t0 = time.time()
    stage_face_ai_router(rows, crops)
    write_rows(out_csv, rows)
    print(f"face/ai/router done in {time.time() - t0:.0f}s", flush=True)
    asyncio.run(stage_experts(rows))
    write_rows(out_csv, rows)
    if args.x2dfd:
        asyncio.run(stage_x2dfd(rows, set(args.x2dfd_groups) if args.x2dfd_groups else None, out_csv))
    stage_decide(rows)
    write_rows(out_csv, rows)
    print(f"saved {out_csv} ({len(rows)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
