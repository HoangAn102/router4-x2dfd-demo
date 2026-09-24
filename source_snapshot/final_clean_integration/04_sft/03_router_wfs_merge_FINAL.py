from pathlib import Path
import pandas as pd
import json
import random
import hashlib
import os

from utils.annotation_utils import (
    compose_labeled_response,
)

BASE = Path("/home/aiotlab/hoangan")
RUN  = BASE/"outputs/ROUTER4_NATIVE_X2DFD_24K_20260915"

ANN   = RUN/"sfs_annotations"
META  = RUN/"train24k_router4_meta.csv"
INDEX = RUN/"annotation_index.json"

OUT   = RUN/"router4_x2dfd_native_sft_24k.json"
AUDIT = RUN/"router_wfs_audit.csv"

LO = 0.30
HI = 0.70
SEED = 20260915

meta = pd.read_csv(META)

meta["image_path"] = meta.image_path.map(
    lambda x: os.path.normpath(str(x))
)

if meta.image_path.nunique() != 24000:
    raise RuntimeError(
        "Router metadata does not contain "
        "24,000 unique paths"
    )

meta_by_path = (
    meta.set_index("image_path")
        .to_dict("index")
)

index = json.load(
    open(INDEX, encoding="utf-8")
)

annotated = {}

for rec in index:

    p = (
        ANN /
        f"{rec['name']}_annotations.json"
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing SFS annotation: {p}"
        )

    data = json.load(
        open(p, encoding="utf-8")
    )

    if len(data) != int(rec["n"]):
        raise RuntimeError(
            f"Bad annotation size: {p}"
        )

    for item in data:

        image = os.path.normpath(
            str(item.get("image"))
        )

        if image in annotated:
            raise RuntimeError(
                f"Duplicate annotation: {image}"
            )

        annotated[image] = item


if len(annotated) != 24000:
    raise RuntimeError(
        f"Expected 24000 annotations, "
        f"got {len(annotated)}"
    )

missing = (
    set(meta_by_path) -
    set(annotated)
)

extra = (
    set(annotated) -
    set(meta_by_path)
)

if missing or extra:
    raise RuntimeError(
        f"Path mismatch: "
        f"missing={len(missing)}, "
        f"extra={len(extra)}"
    )

items = []
audit_rows = []

for image, m in meta_by_path.items():

    ann = annotated[image]

    conv = (
        ann.get("conversations")
        if isinstance(ann, dict)
        else None
    ) or []

    original_answer = ""

    for turn in conv:
        if (
            isinstance(turn, dict)
            and turn.get("from") == "gpt"
        ):
            original_answer = str(
                turn.get("value") or ""
            ).strip()

            break

    if not original_answer:
        raise RuntimeError(
            f"Empty native SFS rationale: {image}"
        )

    label = int(m["label"])

    label_name = (
        "fake"
        if label == 1
        else "real"
    )

    expert = str(
        m["selected_expert"]
    ).strip().lower()

    alias = str(
        m["selected_alias"]
    ).strip()

    if not alias:
        alias = expert.title()

    score = float(
        m["selected_score"]
    )

    # --------------------------------------------------------
    # Human prompt:
    # follows official X2DFD WFS question style,
    # but expert identity is selected PER IMAGE.
    # --------------------------------------------------------

    human = (
        "\nIs this image real or fake? "
        f"And the {alias.lower()} score "
        f"is {score:.3f}."
    )

    # --------------------------------------------------------
    # Preserve original X2DFD SFS rationale.
    # compose_labeled_response is X2DFD's own helper.
    # --------------------------------------------------------

    gpt = compose_labeled_response(
        label_name,
        original_answer,
    )

    # --------------------------------------------------------
    # Reliability-aware Router-WFS.
    #
    # Human ALWAYS receives selected score.
    #
    # GPT support clause only when selected expert
    # strongly agrees with GT, analogous to X2DFD WFS.
    # --------------------------------------------------------

    if label == 1 and score >= HI:

        state = "support"

        gpt += (
            f" Additionally, the selected "
            f"{alias.lower()} expert provides "
            f"strong weak-feature evidence "
            f"supporting fake."
        )

    elif label == 0 and score <= LO:

        state = "support"

        gpt += (
            f" Additionally, the selected "
            f"{alias.lower()} expert provides "
            f"weak-feature evidence supporting real."
        )

    elif (
        (label == 1 and score <= LO)
        or
        (label == 0 and score >= HI)
    ):

        # Important:
        # do NOT fabricate a supporting claim.
        state = "conflict"

    else:

        state = "uncertain"

    items.append({
        "id": "0",
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": human,
            },
            {
                "from": "gpt",
                "value": gpt,
            },
        ],
    })

    audit_rows.append({
        "image_path": image,
        "dataset": m["dataset"],
        "label": label,
        "selected_expert": expert,
        "selected_alias": alias,
        "selected_score": score,
        "wfs_state": state,
        "sfs_rationale_chars":
            len(original_answer),
    })


# ------------------------------------------------------------
# Deterministic shuffle
# ------------------------------------------------------------

rng = random.Random(SEED)
rng.shuffle(items)

for i, item in enumerate(
    items,
    1
):
    item["id"] = str(i)


if len(items) != 24000:
    raise RuntimeError(
        f"Expected 24000 final SFT items, "
        f"got {len(items)}"
    )


with open(
    OUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        items,
        f,
        ensure_ascii=False,
    )


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    AUDIT,
    index=False
)


sha = hashlib.sha256(
    OUT.read_bytes()
).hexdigest()

(
    RUN/"router4_x2dfd_native_sft_24k.sha256"
).write_text(
    f"{sha}  {OUT.name}\n"
)


print("="*72)
print("ROUTER-GUIDED WFS MERGE COMPLETE")
print("="*72)

print("N =",len(items))

print()
print("WFS STATE:")
print(
    audit.wfs_state
         .value_counts()
         .to_string()
)

print()
print("BY EXPERT:")
print(
    pd.crosstab(
        audit.selected_expert,
        audit.wfs_state,
    ).to_string()
)

print()
print("BY DATASET:")
print(
    pd.crosstab(
        audit.dataset,
        audit.wfs_state,
    ).to_string()
)

print()
print("SFT =",OUT)
print("AUDIT =",AUDIT)
print("SHA256 =",sha)
