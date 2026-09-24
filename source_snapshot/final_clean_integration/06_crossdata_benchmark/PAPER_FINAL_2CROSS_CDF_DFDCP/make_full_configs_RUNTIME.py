from pathlib import Path
import copy
import json
import yaml


BASE = Path("/home/aiotlab/hoangan")
X2 = BASE / "projects/X2DFD"

ROOT = (
    BASE /
    "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/smoke/work"
)

SRC = (
    X2 /
    "eval/configs/infer_config.yaml"
)

ORIGINAL = (
    X2 /
    "eval/configs/"
    "infer_full_DFDCP_DFDC_original.yaml"
)

ROUTER = (
    X2 /
    "eval/configs/"
    "infer_full_DFDCP_DFDC_router_moe.yaml"
)


base = yaml.safe_load(
    SRC.read_text(
        encoding="utf-8"
    )
)

inputs = json.loads(
    (
        ROOT /
        "full_inputs.json"
    ).read_text(
        encoding="utf-8"
    )
)


# ------------------------------------------------------------
# ORIGINAL
#
# Keep original weak_supplies unchanged.
# Only replace Tiny_Test inputs with FULL inputs.
# ------------------------------------------------------------

orig = copy.deepcopy(base)

orig["infer"]["inputs"] = inputs

ORIGINAL.write_text(
    yaml.safe_dump(
        orig,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# ROUTER-MOE
# ------------------------------------------------------------

router = copy.deepcopy(base)

router["infer"]["inputs"] = inputs

router["weak_supplies"] = [
    {
        "provider":
            "router_moe_cache",

        "alias":
            "RouterMoE",

        "cache_csv":
            str(
                ROOT /
                "router_moe_cache.csv"
            ),

        # PRIMARY experiment:
        # Preserve expert's original output score.
        "score_col":
            "selected_raw_score",

        "alias_col":
            "selected_alias",

        "thresholds": {
            "lo": 0.30,
            "hi": 0.70,
        },
    }
]


ROUTER.write_text(
    yaml.safe_dump(
        router,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


print("ORIGINAL:")
print(ORIGINAL)

print()
print("ROUTER-MOE:")
print(ROUTER)

print()
print("FULL INPUTS:")

for p in inputs:
    print(" ", p)

print()
print("CONFIGS READY ✅")
