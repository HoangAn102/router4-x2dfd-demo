"""P6 + prompt-format ablation for the X2DFD LLaVA stage (FIX_PLAN PHASE 2).

Loads LLaVA + LoRA once (in-process; stop the web first) and scores the rows of parity_rows.csv
(tools/eval/parity_check.py) under three prompt formats:

  A_live    what the web sent before the audit: X2DFD single_image_infer_with_scores with
            "<image>\\n..." -> conv llava_v0 (adapter dir name has no "v1"), "<im_start><image><im_end>",
            image features inserted; score from calibrators_FINAL_CLEAN
  B_train   exactly what LoRA training tokenized: conv "v1" (--version v1), human text
            "\\nIs this image real or fake? And the {alias} score is {s:.3f}." with NO <image> token
            (the SFT json has none, so LLaVA dropped the image features); score from the v2 calibrator
            (router_teacher_v2/calibrators_train_only.joblib, the one that produced the SFT scores)
  C_v1_img  conv "v1" + "<image>\\n" + image, v2 score (informational only)

For each: generated text (max 4 tokens), step of the first real/fake token, pairwise softmax(real, fake) logit
at that step, verdict by logit vs verdict by text.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

X2ROOT = "/home/aiotlab/hoangan/projects/X2DFD"
sys.path.insert(0, X2ROOT)

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from utils import lora_inference as LI  # noqa: E402

LORA = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
CAL_V2 = "/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib"


def score(tok, model, proc, prompt_text, image_path, use_image):
    dev = model.device
    ids = tokenizer_image_token(prompt_text, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(dev)
    kw = {}
    if use_image:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        kw = dict(images=process_images([img], proc, model.config).to(dev, dtype=torch.float16),
                  image_sizes=[img.size])
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                             max_new_tokens=4, use_cache=True, return_dict_in_generate=True, output_scores=True, **kw)
    gen = out.sequences[0][-len(out.scores):].tolist()
    real_id = tok.encode("real", add_special_tokens=False)[0]
    fake_id = tok.encode("fake", add_special_tokens=False)[0]
    step = next((i for i, t in enumerate(gen) if t in (real_id, fake_id)), None)
    text = tok.decode(gen, skip_special_tokens=True)
    if step is None:
        return dict(text=text, step=-1, fake=np.nan, text_label="none")
    lg = out.scores[step][0].float()
    p = torch.softmax(torch.stack([lg[real_id], lg[fake_id]]), 0)
    return dict(text=text, step=step, fake=float(p[1]), text_label="fake" if gen[step] == fake_id else "real")


def conv_prompt(mode, human):
    c = conv_templates[mode].copy()
    c.append_message(c.roles[0], human)
    c.append_message(c.roles[1], None)
    return c.get_prompt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.rows)
    v2 = joblib.load(CAL_V2)["calibrators"]
    tok, model, proc, _ = LI._get_or_load_model(LORA, BASE)
    res = []
    for _, r in df.iterrows():
        e = r.live_router_expert
        cal_final = float(r[f"live_cal_{e}"])
        cal_v2 = float(v2[e].predict([float(r[f"live_raw_{e}"])])[0])
        q_tail = f"Is this image real or fake? And the {e} score is "
        row = dict(image_path=r.image_path, part=r.part, domain=r.domain, label=int(r.label), expert=e,
                   cal_final=cal_final, cal_v2=cal_v2)
        # A: unchanged live call (X2DFD helper builds the conv itself)
        a = LI.single_image_infer_with_scores(image_path=r.image_path, question=f"<image>\n{q_tail}{cal_final:.3f}.",
                                              model_path=LORA, model_base=BASE, max_new_tokens=4)
        row.update(A_fake=a["fake_score"], A_text=a["answer"])
        b = score(tok, model, proc, conv_prompt("v1", f"\n{q_tail}{cal_v2:.3f}."), r.image_path, False)
        row.update({f"B_{k}": v for k, v in b.items()})
        c = score(tok, model, proc, conv_prompt("v1", f"<image>\n{q_tail}{cal_v2:.3f}."), r.image_path, True)
        row.update({f"C_{k}": v for k, v in c.items()})
        if r.part == "train" and isinstance(r.get("sft_prompt"), str):
            # replay the exact SFT human turn (OOF score from training)
            s = score(tok, model, proc, conv_prompt("v1", r.sft_prompt), r.image_path, False)
            row.update({f"S_{k}": v for k, v in s.items()})
        res.append(row)
        print(len(res), row["label"], e, round(row["A_fake"], 3), round(row["B_fake"], 3), round(row["C_fake"], 3),
              flush=True)
    out = pd.DataFrame(res)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "llava_prompt_parity.csv", index=False)

    from sklearn.metrics import roc_auc_score
    summ = {}
    for part in ("val", "train", "all"):
        d = out if part == "all" else out[out.part == part]
        for v in ("A", "B", "C", "S"):
            col = f"{v}_fake"
            if col not in d or d[col].isna().all():
                continue
            dd = d[d[col].notna()]
            s = dict(n=len(dd), auc=roc_auc_score(dd.label, dd[col]) if dd.label.nunique() > 1 else None,
                     acc=float(((dd[col] >= 0.5).astype(int) == dd.label).mean()))
            if f"{v}_text_label" in dd:
                s["logit_verdict_eq_text"] = float(((dd[col] >= 0.5) == (dd[f"{v}_text_label"] == "fake")).mean())
                s["label_step_counts"] = dd[f"{v}_step"].value_counts().to_dict()
            summ[f"{part}_{v}"] = s
    if "S_text_label" in out:
        t = out[out.part == "train"]
        summ["train_S_reproduces_training_label"] = float(((t.S_text_label == "fake").astype(int) == t.label).mean())
    summ["examples"] = out[[c for c in out.columns if c.endswith("_text")]].head(5).to_dict("records")
    (args.out / "llava_prompt_parity_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
