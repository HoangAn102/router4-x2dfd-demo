"""P7: web face crop vs the historical DeepfakeBench crops used in training (FIX_PLAN PHASE 2).

For N FF++ c23 videos (half real, half Deepfakes/Face2Face/FaceSwap/NeuralTextures), every frame index that
has a historical crop in FFPP_X2DFD32_CROPPED/<class>/<video>/<idx>.png is decoded the way DeepfakeBench
facecrop() does (sequential cv2.VideoCapture.read) and cropped with the web's FaceCropper
(crop_image_with_detector, PIL RGB in), then compared to the stored PNG with SSIM (+ mean abs pixel diff).

    python tools/eval/facecrop_parity.py --out DIR [--videos 10] [--frames 4]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.backend.app.services.face_cropper import FaceCropper  # noqa: E402

FF = Path("/home/aiotlab/hoangan/datasets/FaceForensicsC23")
CROPS = FF / "FFPP_X2DFD32_CROPPED"
VIDEOS = FF / "FaceForensics++_C23"
FAKES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def pick(n, seed=42):
    rng = random.Random(seed)
    real = sorted(p for p in (CROPS / "real").iterdir() if p.is_dir() and any(p.glob("*.png")))
    out = [("real", d, VIDEOS / "real" / f"{d.name.split('__')[-1]}.mp4") for d in rng.sample(real, n // 2)]
    for i in range(n - n // 2):
        m = FAKES[i % len(FAKES)]
        d = rng.choice(sorted(p for p in (CROPS / m).iterdir() if p.is_dir() and any(p.glob("*.png"))))
        out.append((m, d, VIDEOS / "fake" / m / f"{d.name.split('__')[-1]}.mp4"))
    return out


def frames(video, idxs):
    cap = cv2.VideoCapture(str(video))
    got, i = {}, 0
    while len(got) < len(idxs):
        ok, f = cap.read()
        if not ok:
            break
        if i in idxs:
            got[i] = f
        i += 1
    cap.release()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos", type=int, default=10)
    ap.add_argument("--frames", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, cdir, video in pick(a.videos):
        stored = sorted(cdir.glob("*.png"))
        stored = [stored[i] for i in np.linspace(0, len(stored) - 1, a.frames, dtype=int)]
        idx = {int(p.stem): p for p in stored}
        fr = frames(video, set(idx))
        for i, p in idx.items():
            ref = cv2.imread(str(p))
            r = dict(cls=cls, video=video.name, frame=i, stored=str(p))
            if i not in fr:
                r.update(status="frame_missing")
            else:
                found = FaceCropper.crop_image_with_detector(Image.fromarray(cv2.cvtColor(fr[i], cv2.COLOR_BGR2RGB)))
                if found is None:
                    r.update(status="no_face")
                else:
                    web = cv2.cvtColor(np.asarray(found[0]), cv2.COLOR_RGB2BGR)
                    r.update(status="ok", detector=found[1],
                             ssim=float(structural_similarity(ref, web, channel_axis=2)),
                             mad=float(np.abs(ref.astype(int) - web.astype(int)).mean()),
                             identical=bool(np.array_equal(ref, web)))
                    cv2.imwrite(str(a.out / f"{video.stem}_{cls}_{i:03d}_web.png"), web)
            rows.append(r)
            print(r, flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    summ = dict(n=len(rows), ok=len(ok), no_face=sum(r["status"] == "no_face" for r in rows),
                frame_missing=sum(r["status"] == "frame_missing" for r in rows),
                ssim_min=min(r["ssim"] for r in ok), ssim_mean=float(np.mean([r["ssim"] for r in ok])),
                ssim_ge_095=sum(r["ssim"] >= 0.95 for r in ok), identical=sum(r["identical"] for r in ok),
                mad_max=max(r["mad"] for r in ok))
    (a.out / "facecrop_parity.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
