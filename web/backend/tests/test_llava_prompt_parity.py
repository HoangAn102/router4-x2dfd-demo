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
BASE_LLAVA = Path("/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b")
HAS_LLAVA = importlib.util.find_spec("llava") is not None and importlib.util.find_spec("transformers") is not None


class PromptStringTests(unittest.TestCase):
    def test_no_image_token_and_leading_newline(self):
        p = X2DFDClient.build_wfs_prompt("Texture", 1.0)
        self.assertEqual(p, "\nIs this image real or fake? And the texture score is 1.000.")
        self.assertNotIn("<image>", p)

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


if __name__ == "__main__":
    unittest.main()
