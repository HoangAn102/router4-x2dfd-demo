#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

from pathlib import Path


DEFAULT_X2ROOT = "/home/aiotlab/hoangan/projects/X2DFD"


# ---------------------------------------------------------------------------
# Train/inference parity (audit 2026-10-30, AUDIT.md P4-P6).
#
# The LoRA router4_x2dfd_FINAL_CLEAN was trained by LLaVA train.py with
#   --version v1                         -> conversation template conv_templates["v1"]
#   human turn "\nIs this image real or fake? And the {alias} score is {s:.3f}."
#                                        -> no "<image>" token in any of the 24,000 SFT
#                                           samples, so LLaVA dropped the image features
#                                           (llava_arch: image_features[0:0]).
# X2DFD's single_image_infer_with_scores() instead picks conv "llava_v0" (adapter dir
# name has no "v1") and prepends "<im_start><image><im_end>\n" plus image features, a
# token sequence the LoRA never saw. Here the prompt is tokenized exactly as in training;
# model loading and the real/fake logit read-out are X2DFD's.
# ---------------------------------------------------------------------------

TRAIN_CONV_TEMPLATE = "v1"


def _x2root():

    x2root = Path(
        os.environ.get(
            "X2DFD_PROJECT_ROOT",
            DEFAULT_X2ROOT,
        )
    ).resolve()

    if not x2root.exists():
        raise RuntimeError(
            f"X2DFD root missing: {x2root}"
        )

    if str(x2root) not in sys.path:
        sys.path.insert(0, str(x2root))

    return x2root


def build_train_prompt(question):
    """Full LLaVA prompt string, identical to the prompt part LoRA training tokenized."""

    _x2root()

    from llava.conversation import conv_templates

    if "<image>" in question:
        raise ValueError(
            "SFT prompts contain no <image> token; "
            "refusing a prompt the LoRA never saw"
        )

    conv = conv_templates[TRAIN_CONV_TEMPLATE].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()


def _generate(question, model_path, model_base, max_new_tokens, with_scores):

    _x2root()

    import torch

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from utils.lora_inference import _get_or_load_model

    tokenizer, model, _, _ = _get_or_load_model(
        model_path,
        model_base,
    )

    input_ids = tokenizer_image_token(
        build_train_prompt(question),
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max(int(max_new_tokens), 1),
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=with_scores,
        )

    return tokenizer, out


def score_with_train_prompt(question, *, model_path, model_base, max_new_tokens=4):
    """REAL/FAKE pairwise softmax of the logits at the first generated real/fake token
    (same read-out as X2DFD X2DFD_CONTINUOUS_SCORE_V3)."""

    import torch

    tokenizer, out = _generate(
        question,
        model_path,
        model_base,
        max_new_tokens,
        True,
    )

    generated = out.sequences[0][-len(out.scores):].tolist()
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    real_id = tokenizer.encode("real", add_special_tokens=False)
    fake_id = tokenizer.encode("fake", add_special_tokens=False)

    if len(real_id) != 1 or len(fake_id) != 1:
        raise RuntimeError(
            f"real/fake are not single tokens: {real_id} {fake_id}"
        )

    real_id, fake_id = real_id[0], fake_id[0]

    step = next(
        (
            i
            for i, t in enumerate(generated)
            if t in (real_id, fake_id)
        ),
        None,
    )

    if step is None:
        raise RuntimeError(
            "Generated response contains no real/fake token; "
            f"answer={answer!r}"
        )

    logits = out.scores[step][0].float()
    pair = torch.softmax(
        torch.stack([logits[real_id], logits[fake_id]]),
        dim=0,
    )

    return {
        "answer": answer,
        "label_step": step,
        "real_score": round(float(pair[0]), 8),
        "fake_score": round(float(pair[1]), 8),
    }


def explain_with_train_prompt(question, *, model_path, model_base, max_new_tokens=512):

    tokenizer, out = _generate(
        question,
        model_path,
        model_base,
        max_new_tokens,
        False,
    )

    return tokenizer.batch_decode(
        out.sequences,
        skip_special_tokens=True,
    )[0].strip()


def _load_infer_function():

    return (
        score_with_train_prompt,
        explain_with_train_prompt,
    )


def _run_one(
    score_fn,
    explain_fn,
    *,
    mode,
    image,
    prompt,
    lora_dir,
    base_model,
    max_new_tokens,
):

    image_path = (
        Path(image)
        .expanduser()
        .resolve(strict=True)
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image missing: {image_path}"
        )

    # stdout stays JSON-only.
    with contextlib.redirect_stdout(
        sys.stderr
    ):

        if mode == "score":

            # The image is validated above but not tokenized: the LoRA was trained
            # without image tokens (see build_train_prompt). Router4 and the
            # experts are the image-dependent stages.
            result = score_fn(
                prompt,
                model_path=lora_dir,
                model_base=base_model,
                max_new_tokens=max_new_tokens,
            )

            return {
                "ok": True,
                "mode": "score",
                "answer": result.get("answer"),
                "label_step": result.get("label_step"),
                "real_score":
                    result.get("real_score"),
                "fake_score":
                    result.get("fake_score"),
            }

        if mode == "explain":

            # Greedy continuation of the same training prompt, returned unmodified.
            answer = explain_fn(
                prompt,
                model_path=lora_dir,
                model_base=base_model,
                max_new_tokens=max_new_tokens,
            )

            return {
                "ok": True,
                "mode": "explain",
                "answer": answer,
            }

        raise ValueError(
            f"Unknown X2DFD mode: {mode}"
        )


def server_mode(args):

    score_fn, explain_fn = _load_infer_function()

    protocol_out = sys.stdout

    print(
        "[X2DFD_SERVER] READY",
        file=sys.stderr,
        flush=True,
    )

    for raw in sys.stdin:

        raw = raw.strip()

        if not raw:
            continue

        try:
            req = json.loads(raw)

            response = _run_one(
                score_fn,
                explain_fn,

                mode=req.get(
                    "mode",
                    "score",
                ),

                image=req["image"],
                prompt=req["prompt"],

                lora_dir=(
                    req.get("lora_dir")
                    or args.lora_dir
                ),

                base_model=(
                    req.get("base_model")
                    or args.base_model
                ),

                max_new_tokens=int(
                    req.get(
                        "max_new_tokens",
                        args.max_new_tokens,
                    )
                ),
            )

        except Exception as e:

            traceback.print_exc(
                file=sys.stderr
            )

            response = {
                "ok": False,
                "error_type": type(e).__name__,
                "error": str(e),
            }

        protocol_out.write(
            json.dumps(
                response,
                ensure_ascii=False,
            ) + "\n"
        )

        protocol_out.flush()


def oneshot_mode(args):

    score_fn, explain_fn = _load_infer_function()

    try:
        result = _run_one(
            score_fn,
            explain_fn,
            mode=args.mode,
            image=args.image,
            prompt=args.prompt,
            lora_dir=args.lora_dir,
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
            )
        )

    except Exception as e:

        print(
            f"X2DFD inference failed: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )

        raise SystemExit(1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server",
        action="store_true",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "score",
            "explain",
        ],
        default="score",
    )

    parser.add_argument("--image")
    parser.add_argument("--prompt")

    parser.add_argument(
        "--lora-dir",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.server:
        server_mode(args)
        return

    if not args.image or not args.prompt:
        parser.error(
            "--image and --prompt required "
            "outside --server mode"
        )

    oneshot_mode(args)


if __name__ == "__main__":
    main()
