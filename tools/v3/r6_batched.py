"""R6 (full): batched, teacher-forced real/fake read-out for the image-conditioned LLaVA systems (V3, bold).

Every SFT answer (and every greedy answer seen so far) starts with the same words before the label token
("This image is real/fake"). The single-image path (generate, greedy, read logits at the first real/fake step) is
therefore equal to ONE forward pass over prompt + that fixed prefix, reading the logits at the last position.
This script does that forward for a batch of right-padded prompts (LLaVA's native padding), and checks per sample
that the greedy argmax at every prefix position IS the forced token; samples that fail the check are re-scored
with the original single-image function. `--verify N` first scores N frames with the single-image function, derives
the prefix from them and reports |delta fake_prob| between the two paths (mean, p99, max; logged in PROGRESS).

  V3    prompt = conv v1 (training template) via web/model_worker/x2dfd_worker_cli.build_train_prompt,
        reference = x2dfd_worker_cli.score_with_train_prompt
  bold  prompt = X2DFD utils.lora_inference.single_image_infer_with_scores construction (conv picked from the
        adapter name, <im_start><image><im_end>), reference = that function
  a     X2DFD original: author's questions (r6_author_questions.py = eval.infer.runner's own question builder),
        author adapter, same prompt construction/reference as bold (the runner calls single_image_infer_with_scores)

    x2python tools/v3/r6_batched.py --frames frames_full.csv --system V3 --lora DIR --out results_V3.csv
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "web/model_worker"))
sys.path.insert(0, "/home/aiotlab/hoangan/projects/X2DFD")

import x2dfd_worker_cli as W  # noqa: E402
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from utils import lora_inference as LI  # noqa: E402

BASE = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"
LORA_AUTHOR = "/home/aiotlab/hoangan/projects/X2DFD/weights/_official_extract/weight/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]"
LORA_FC = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"


def q(expert, cal):
    return f"<image>\nIs this image real or fake? And the {expert} score is {cal:.3f}."


def bold_prompt(question, model_path):
    """Same string single_image_infer_with_scores builds (conv mode from the adapter name)."""
    name = LI.get_model_name_from_path(model_path).lower()
    mode = next((m for k, m in (("llama-2", "llava_llama_2"), ("mistral", "mistral_instruct"),
                                ("v1.6-34b", "chatml_direct"), ("v1", "llava_v1"), ("mpt", "mpt")) if k in name),
                "llava_v0")
    qs = re.sub("<image>", DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN, question)
    conv = conv_templates[mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class Frames(torch.utils.data.Dataset):
    def __init__(self, rows, proc, cfg):
        self.rows, self.proc, self.cfg = rows, proc, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, question = self.rows[i]
        img = Image.open(path).convert("RGB")
        return path, question, process_images([img], self.proc, self.cfg)[0], img.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--system", required=True, choices=["V3", "bold", "a"])
    ap.add_argument("--lora")
    ap.add_argument("--questions", type=Path, help="system a: r6_author_questions.py output")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="test,dev")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", type=int, default=64)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--reverify", action="store_true", help="re-run the batched-vs-single comparison (cached refs)")
    a = ap.parse_args()

    lora = a.lora or (LORA_AUTHOR if a.system == "a" else LORA_FC)
    df = pd.read_csv(a.frames)
    df = df[df.split.isin(a.splits.split(","))]
    if a.system == "V3":
        df = df.assign(question=[q(e, c) for e, c in zip(df.V3_expert, df.V3_cal)])
        prompt = W.build_train_prompt

        def single(path, question):
            return W.score_with_train_prompt(question, model_path=lora, model_base=BASE, max_new_tokens=4,
                                             image_path=path)
    else:
        if a.system == "a":
            qa = pd.read_csv(a.questions).drop_duplicates("image_path").set_index("image_path").question
            df = df.assign(question=df.image_path.map(qa))
            assert df.question.notna().all(), "author questions missing for some frames"
        else:
            df = df.assign(question=[q(e, c) for e, c in zip(df.b_expert, df.bold_cal)])

        def prompt(question):
            return bold_prompt(question, lora)

        def single(path, question):
            return LI.single_image_infer_with_scores(image_path=path, question=question, model_path=lora,
                                                     model_base=BASE, max_new_tokens=4)

    tok, model, proc, _ = LI._get_or_load_model(lora, BASE)
    dev, dtype = model.device, next(model.parameters()).dtype
    real_id = tok.encode("real", add_special_tokens=False)[0]
    fake_id = tok.encode("fake", add_special_tokens=False)[0]
    out_cols = ["image_path", f"{a.system}_fake", f"{a.system}_path"]
    meta = a.out.with_suffix(".meta.json")

    # 1) reference path on the first frames: forced prefix + agreement of the two read-outs
    if meta.exists():
        info = json.loads(meta.read_text())
    else:
        ref = pd.concat([g.head(max(1, a.verify // df.dataset.nunique())) for _, g in df.groupby("dataset")])
        prefixes, ref_scores = set(), {}
        for r in ref.itertuples():
            res = single(r.image_path, r.question)
            ref_scores[r.image_path] = res["fake_score"]
            prefixes.add(re.split(r" (?:real|fake)\b", res["answer"])[0])
        if len(prefixes) != 1:
            raise SystemExit(f"greedy prefix not constant: {prefixes}")
        info = dict(prefix=prefixes.pop(), ref=ref_scores)
    prefix_ids = tok(info["prefix"], add_special_tokens=False).input_ids

    def encode(question):
        return tokenizer_image_token(prompt(question), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").tolist() + prefix_ids

    @torch.inference_mode()
    def score_batch(batch):
        paths, questions, imgs, sizes = zip(*batch)
        seqs = [encode(x) for x in questions]
        n = max(map(len, seqs))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        # Same computation as model(...).logits, but the LM head is applied only at the positions read out
        # (lm_head is position-wise), which keeps a batch well under the 16 GB per-job budget.
        _, pos, am, _, emb, _ = model.prepare_inputs_labels_for_multimodal(
            ids.to(dev), None, mask.to(dev), None, None, torch.stack(imgs).to(dev, dtype), list(sizes))
        hidden = model.model(inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=False).last_hidden_state
        lens = am.sum(1).tolist()
        res = []
        m = len(prefix_ids)
        for i, s in enumerate(seqs):
            last = len(s) - 1 + (s.count(IMAGE_TOKEN_INDEX) * (model.get_vision_tower().num_patches - 1))
            assert last == lens[i] - 1, (last, lens[i])
            lg = model.lm_head(hidden[i, last - m:last + 1]).float()  # rows: prefix predictions, then the label
            ok = all(int(lg[j].argmax()) == prefix_ids[j] for j in range(m)) and int(lg[m].argmax()) in (real_id, fake_id)
            pair = torch.softmax(torch.stack([lg[m, real_id], lg[m, fake_id]]), 0)
            res.append((paths[i], round(float(pair[1]), 8), ok, questions[i]))
        return res

    if not meta.exists() or a.reverify:
        qmap = dict(zip(df.image_path, df.question))
        ds = Frames([(p, qmap[p]) for p in info["ref"]], proc, model.config)
        got = [r for k in range(0, len(ds), a.bs) for r in score_batch([ds[i] for i in range(k, min(k + a.bs, len(ds)))])]
        torch.cuda.empty_cache()
        diffs = [abs(f - info["ref"][p]) for p, f, ok, _ in got]
        worst = max(range(len(got)), key=lambda i: diffs[i])
        info.update(n_verify=len(got), max_abs_diff=max(diffs), mean_abs_diff=sum(diffs) / len(diffs),
                    worst=dict(path=got[worst][0], batched=got[worst][1], single=info["ref"][got[worst][0]]),
                    p99_abs_diff=sorted(diffs)[int(0.99 * (len(diffs) - 1))],
                    prefix_check_ok=sum(ok for *_, ok, _ in got),
                    verdict_agree=sum((f >= 0.5) == (info["ref"][p] >= 0.5) for p, f, *_ in got))
        meta.write_text(json.dumps(info, indent=1))
        pd.DataFrame(dict(image_path=[g[0] for g in got], batched=[g[1] for g in got],
                          single=[info["ref"][g[0]] for g in got])).to_csv(meta.with_suffix(".verify.csv"), index=False)
        print({k: v for k, v in info.items() if k != "ref"}, flush=True)
        if a.verify_only:
            return
        # fp16 noise between incremental (KV-cache) decoding and one prefill pass: unbiased, largest near p~0.5-0.8.
        # Criterion (logged, revised after the 60-frame check gave max 0.006 at bs 16): every verdict equal,
        # mean |delta| <= 1e-3 and p99 |delta| <= 5e-3; max |delta| is reported.
        if info["mean_abs_diff"] > 1e-3 or info["p99_abs_diff"] > 5e-3 or info["verdict_agree"] != len(got):
            raise SystemExit("batched read-out does not match the single-image path; not used")

    # 2) all frames, resumable
    done = set(pd.read_csv(a.out).image_path) if a.out.exists() else set()
    todo = df[~df.image_path.isin(done)].drop_duplicates("image_path")
    loader = torch.utils.data.DataLoader(Frames(list(zip(todo.image_path, todo.question)), proc, model.config),
                                         batch_size=a.bs, num_workers=a.workers, collate_fn=list)
    t0, k, fb = time.time(), 0, 0
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if not done:
            w.writerow(out_cols)
        for batch in loader:
            for p, f, ok, question in score_batch(batch):
                if not ok:
                    f, fb = single(p, question)["fake_score"], fb + 1
                w.writerow([p, f, "batched" if ok else "single"])
            k += len(batch)
            if k % (a.bs * 50) < a.bs:
                fh.flush()
                print(f"{k}/{len(todo)} {k / (time.time() - t0):.1f} img/s fallback={fb}", flush=True)
    print("done", k, "fallback", fb, flush=True)


if __name__ == "__main__":
    main()
