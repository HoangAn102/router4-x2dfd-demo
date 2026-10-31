"""R7 (6): (a) X2DFD original, (b) parity v2 and V3 on every ~/hoangan/test_imgs image (+ the two girl originals).
REPORT ONLY: nothing here chooses a parameter. Labels are the documented truths (eval_runs/20261026 manifest).

  prep   web face crop (dlib -> YuNet -> Haar, as live), raw scores of all 4 experts, Router v2 / V3 choice,
         calibrated score, (b) from the teacher-forced score table  -> DIR/frames_full.csv (r6_* format)
  table  merge DIR/results_{a,V3}.csv (r6_batched.py on DIR/frames_full.csv) -> DIR/test_imgs_table.{csv,md}

    x2python tools/v3/r7_test_imgs.py prep --out DIR
    (cd X2DFD; x2python tools/v3/r6_author_questions.py --frames DIR/frames_full.csv --out DIR/a_questions.csv)
    x2python tools/v3/r6_batched.py --frames DIR/frames_full.csv --system V3 --lora LORA_V3 --out DIR/results_V3.csv
    x2python tools/v3/r6_batched.py --frames DIR/frames_full.csv --system a --questions DIR/a_questions.csv --out DIR/results_a.csv
    x2python tools/v3/r7_test_imgs.py table --out DIR
"""
import argparse
import asyncio
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools/eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

H = Path("/home/aiotlab/hoangan")
IMAGES = sorted((H / "test_imgs").iterdir()) + [H / "01_AI_BE_GAI_OM_CHO_CON.png",
                                                 H / "eval_runs/20261026/mustpass/girl_q90.jpg"]
# truth: phone photos (2aOboR*) are real; MAO_DANH_*, 03_ANH_AI_*, FAKE_*, the girl (PNG/JPG/Zalo z590*) are fake
REAL_PREFIX = ("2aOboR",)


def prep(out):
    from build_sft_v3 import route_batch
    from eval_r6 import SYS
    from r6_frames import EXPERTS, TABLE_TF, cal_selected
    from web.backend.app.services.face_cropper import FaceCropper
    from web.model_worker.expert_client import ExpertClient

    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, src in enumerate(IMAGES):
        dst = out / "crops" / f"{i:02d}_{src.stem[:40]}.png"
        try:
            _, det = FaceCropper.crop_face_file_with_detector(src, dst)
        except Exception as e:  # NO_FACE_DETECTED (MediaValidationError) is a result, not a failure
            det, dst = f"no_face:{type(e).__name__}", None
        rows.append(dict(split="test", dataset="test_imgs", video=str(src), source=str(src),
                         label=0 if src.name.startswith(REAL_PREFIX) else 1, detector=det,
                         image_path=None if dst is None else str(dst)))
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "images.csv", index=False)
    df = meta[meta.image_path.notna()].reset_index(drop=True)
    ec = ExpertClient()
    for e in EXPERTS:
        paths = [Path(p) for p in df.image_path]
        res = asyncio.run(ec.compute_expert_scores_batch(e, paths))
        df[f"score_{e}"] = [res[str(p.resolve())][0] for p in paths]
    for name in ("b", "V3"):
        ckpt, calp, key = SYS[name]
        df[f"{name}_expert"] = route_batch(df.image_path.tolist(), ckpt)
        df[f"{name}_cal"] = cal_selected(df, f"{name}_expert", joblib.load(calp)[key])
    t = pd.read_csv(TABLE_TF)
    tab = {(al, f"{s:.3f}"): p for al, s, p in zip(t.alias, t.score_3dp, t.fake_prob)}
    df["b_fake"] = [tab[(e, f"{c:.3f}")] for e, c in zip(df.b_expert, df.b_cal)]
    df.to_csv(out / "frames_full.csv", index=False)
    print(df[["image_path", "detector", "b_expert", "V3_expert", "score_blending", "score_diffusion", "b_fake"]])


def table(out):
    meta = pd.read_csv(out / "images.csv")
    df = pd.read_csv(out / "frames_full.csv")
    for s in ("a", "V3"):
        r = pd.read_csv(out / f"results_{s}.csv").drop_duplicates("image_path").set_index("image_path")
        df[f"{s}_fake"] = df.image_path.map(r[f"{s}_fake"])
        df[f"{s}_path"] = df.image_path.map(r[f"{s}_path"])
    df = meta[["source", "label", "detector"]].merge(df.drop(columns=["label", "detector"]), on="source", how="left")
    v = lambda p: "" if pd.isna(p) else ("FAKE" if p >= 0.5 else "REAL")
    for s in ("a", "b", "V3"):
        df[f"{s}_verdict"] = df[f"{s}_fake"].map(v)
        df[f"{s}_ok"] = [None if not vv else (vv == ("FAKE" if y else "REAL"))
                         for vv, y in zip(df[f"{s}_verdict"], df.label)]
    df["all_wrong"] = [all(x is False for x in (a, b, c)) for a, b, c in zip(df.a_ok, df.b_ok, df.V3_ok)]
    df.to_csv(out / "test_imgs_table.csv", index=False)
    f = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    lines = ["| Ảnh | Nhãn | Crop | (a) tác giả | (b) | V3 | Router b / V3 | Blending raw | Diffusion raw |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        cell = lambda s: "NO_FACE" if pd.isna(getattr(r, "image_path")) else \
            f"{getattr(r, s + '_verdict')} {f(getattr(r, s + '_fake'))}" + ("" if getattr(r, s + "_ok") else " ✗")
        lines.append(f"| {Path(r.source).name[:34]} | {'FAKE' if r.label else 'REAL'} | {r.detector} | {cell('a')} | "
                     f"{cell('b')} | {cell('V3')} | {r.b_expert if isinstance(r.b_expert, str) else '—'} / "
                     f"{r.V3_expert if isinstance(r.V3_expert, str) else '—'} | {f(r.score_blending)} | "
                     f"{f(r.score_diffusion)} |")
    (out / "test_imgs_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("all three wrong:", [Path(s).name for s in df.source[df.all_wrong]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "table"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"prep": prep, "table": table}[a.stage](a.out)
