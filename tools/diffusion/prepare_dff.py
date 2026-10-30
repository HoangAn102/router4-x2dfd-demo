"""TRAIN_DIFFUSION_PLAN D0.2: DeepFakeFace (OpenRL, HF, openrail) -> identity-disjoint split + web face crop.

Subsets (raw/<subset>/<xx>/<wikiid>_<dob>_<year>.jpg, IMDB-WIKI layout):
  wiki        real (IMDB-WIKI)
  text2img    fake, Stable Diffusion v1.5
  inpainting  fake, Stable Diffusion inpainting (of the matching wiki photo)
Split by IMDB-WIKI person id (same id in all three subsets): sha1(id) % 100 < 70 train, < 85 val, else test,
so a person (and an inpainting fake and its real source photo) never crosses splits.
Crop = the web's FaceCropper with FACE_CROP_BACKEND=dlib (DeepfakeBench extract_aligned_face_dlib, 1.3x, 256),
saved like the web saves model inputs. Images without a detected face are recorded and dropped.
Output manifest columns follow MAX_CLEAN_TRAIN.csv (image_path,label,domain,generator,dataset,class_name,
original_path,split); expert/teacher columns are filled by later steps.

    FACE_CROP_BACKEND=dlib RUN_MODE=live python tools/diffusion/prepare_dff.py --root DIR [--per-subset N]
"""
import argparse
import csv
import hashlib
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SUBSETS = {"wiki": (0, "real"), "text2img": (1, "sd15_text2img"), "inpainting": (1, "sd_inpainting")}
FIELDS = ["image_path", "label", "domain", "generator", "dataset", "class_name", "original_path", "split",
          "person_id"]


def split_of(pid):
    h = int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def crop_one(job):
    src, dst = job
    from PIL import Image, ImageOps
    from web.backend.app.services.face_cropper import FaceCropper
    if Path(dst).exists():
        return src, dst, "ok"
    try:
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        found = FaceCropper.crop_image_with_detector(img)
    except Exception as exc:  # unreadable file
        return src, None, f"error:{type(exc).__name__}"
    if found is None:
        return src, None, "no_face"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    FaceCropper.save_model_image(found[0], Path(dst))
    return src, dst, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--per-subset", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert os.environ.get("FACE_CROP_BACKEND") == "dlib", "run with FACE_CROP_BACKEND=dlib (training crop)"

    jobs, meta = [], {}
    for sub, (label, gen) in SUBSETS.items():
        files = sorted((a.root / "raw" / sub).rglob("*.jpg"))
        rng = random.Random(42)
        files = rng.sample(files, min(a.per_subset, len(files)))
        for f in files:
            dst = a.root / "crops" / sub / f.parent.name / (f.stem + ".png")
            pid = f.stem.split("_")[0]
            jobs.append((str(f), str(dst)))
            meta[str(f)] = dict(label=label, domain="DFF_Diffusion", generator=gen, dataset="DeepFakeFace",
                                class_name=sub, original_path=str(f), split=split_of(pid), person_id=pid)
    rows, status = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (src, dst, st) in enumerate(ex.map(crop_one, jobs, chunksize=32)):
            status[st] = status.get(st, 0) + 1
            if dst:
                rows.append(dict(image_path=dst, **meta[src]))
            if (i + 1) % 2000 == 0:
                print(i + 1, status, flush=True)
    for split in ("train", "val", "test"):
        with open(a.root / f"DFF_{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(r for r in rows if r["split"] == split)
    with open(a.root / "crop_status.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["status", "count"])
        w.writerows(sorted(status.items()))
    print("done", status)


if __name__ == "__main__":
    main()
