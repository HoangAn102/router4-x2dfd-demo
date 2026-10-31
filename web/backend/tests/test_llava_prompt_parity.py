"""The live LLaVA prompt must tokenize exactly like the LoRA's SFT samples (audit 2026-10-30, P4/P5).

The token-level test needs llava + transformers (X2DFD env):
    /home/aiotlab/miniconda3/envs/X2DFD/bin/python -m unittest web.backend.tests.test_llava_prompt_parity
"""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

from web.model_worker.x2dfd_client import X2DFDClient

SFT_JSON = Path("/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/04_sft/"
                "router4_x2dfd_FINAL_CLEAN_SFT.json")
V3_SFT_JSON = Path("/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030/05_sft/router4_x2dfd_V3_SFT.json")
BASE_LLAVA = Path("/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b")
HAS_LLAVA = importlib.util.find_spec("llava") is not None and importlib.util.find_spec("transformers") is not None


class PromptStringTests(unittest.TestCase):
    def test_no_image_token_and_leading_newline(self):
        p = X2DFDClient.build_wfs_prompt("Texture", 1.0)
        self.assertEqual(p, "\nIs this image real or fake? And the texture score is 1.000.")
        self.assertNotIn("<image>", p)

    def test_image_prefix_when_lora_trained_with_images(self):
        from unittest import mock
        from web.backend.app.config import settings
        with mock.patch.object(settings, "LORA_PROMPT_HAS_IMAGE", True):
            p = X2DFDClient.build_wfs_prompt("Diffusion", 0.5)
        self.assertEqual(p, "<image>\nIs this image real or fake? And the diffusion score is 0.500.")

    @unittest.skipUnless(SFT_JSON.exists(), "SFT json not on this machine")
    def test_matches_sft_human_turns(self):
        import re
        items = json.load(open(SFT_JSON))[:500]
        for it in items:
            human = it["conversations"][0]["value"]
            alias, score = re.fullmatch(r"\nIs this image real or fake\? And the (\w+) score is ([0-9.]+)\.",
                                        human).groups()
            self.assertEqual(X2DFDClient.build_wfs_prompt(alias.title(), float(score)), human)


@unittest.skipUnless(HAS_LLAVA and SFT_JSON.exists() and BASE_LLAVA.exists(), "needs X2DFD env + artifacts")
class TokenParityTests(unittest.TestCase):
    def test_input_ids_equal_training_prompt_ids(self):
        import transformers
        from llava import conversation as conv_lib
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        from llava.train import train as T

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_worker"))
        from x2dfd_worker_cli import build_train_prompt

        tok = transformers.AutoTokenizer.from_pretrained(str(BASE_LLAVA), use_fast=False, model_max_length=2048,
                                                         padding_side="right")
        tok.pad_token = tok.unk_token
        T.conversation_lib.default_conversation = conv_lib.conv_templates["v1"]  # --version v1

        class DA:
            is_multimodal = True
            mm_use_im_start_end = False

        for it in json.load(open(SFT_JSON))[:20]:
            out = T.preprocess(T.preprocess_multimodal(copy.deepcopy([it["conversations"]]), DA()), tok,
                               has_image=True)
            ids, labels = out["input_ids"][0], out["labels"][0]
            train_prompt = ids[:int((labels != -100).nonzero()[0])].tolist()
            live = tokenizer_image_token(build_train_prompt(it["conversations"][0]["value"]), tok,
                                         IMAGE_TOKEN_INDEX)
            self.assertEqual(live, train_prompt)
            self.assertNotIn(IMAGE_TOKEN_INDEX, live)

            # V3 SFT format (author's "<image>\n" prefix): one image token, same position as training
            conv = copy.deepcopy(it["conversations"])
            conv[0]["value"] = "<image>" + conv[0]["value"]
            out = T.preprocess(T.preprocess_multimodal([conv], DA()), tok, has_image=True)
            ids, labels = out["input_ids"][0], out["labels"][0]
            train_prompt = ids[:int((labels != -100).nonzero()[0])].tolist()
            live = tokenizer_image_token(build_train_prompt(conv[0]["value"]), tok, IMAGE_TOKEN_INDEX)
            self.assertEqual(live, train_prompt)
            self.assertEqual(live.count(IMAGE_TOKEN_INDEX), 1)


@unittest.skipUnless(HAS_LLAVA and V3_SFT_JSON.exists() and BASE_LLAVA.exists(), "needs X2DFD env + artifacts")
class ExplainParityTests(unittest.TestCase):
    """R7 (1): the explanation (LoRA V3) is generated from exactly the training prompt ids (conv v1, one <image>),
    not through X2DFD single_image_infer (which picks llava_v0 from the adapter dir name and adds <im_start>)."""

    def test_explain_input_ids_equal_v3_training_prompt(self):
        from unittest import mock

        import torch
        import transformers
        from llava import conversation as conv_lib
        from llava.train import train as T

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_worker"))
        import x2dfd_worker_cli as W
        W._x2root()  # puts the X2DFD root on sys.path
        from utils import lora_inference as LI

        tok = transformers.AutoTokenizer.from_pretrained(str(BASE_LLAVA), use_fast=False, model_max_length=2048,
                                                         padding_side="right")
        tok.pad_token = tok.unk_token
        T.conversation_lib.default_conversation = conv_lib.conv_templates["v1"]

        class DA:
            is_multimodal = True
            mm_use_im_start_end = False

        seen = {}

        class FakeModel:
            device = torch.device("cpu")
            config = None

            def parameters(self):
                return iter([torch.zeros(1, dtype=torch.float16)])

            def generate(self, input_ids, **kw):
                seen["ids"], seen["kw"] = input_ids[0].tolist(), kw
                return mock.Mock(sequences=torch.tensor([[tok.convert_tokens_to_ids("▁This")]]))

        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "x.png"
            Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
            with mock.patch.object(LI, "_get_or_load_model", return_value=(tok, FakeModel(), None, None)), \
                    mock.patch("llava.mm_utils.process_images", return_value=torch.zeros(1, 3, 336, 336)):
                for it in json.load(open(V3_SFT_JSON))[:20]:
                    out = T.preprocess(T.preprocess_multimodal(copy.deepcopy([it["conversations"]]), DA()), tok,
                                       has_image=True)
                    ids, labels = out["input_ids"][0], out["labels"][0]
                    train_prompt = ids[:int((labels != -100).nonzero()[0])].tolist()
                    W.explain_with_train_prompt(it["conversations"][0]["value"], model_path="V3", model_base="b",
                                                image_path=str(img))
                    self.assertEqual(seen["ids"], train_prompt)
                    self.assertIn("images", seen["kw"])  # image features are fed, as in training


if __name__ == "__main__":
    unittest.main()
