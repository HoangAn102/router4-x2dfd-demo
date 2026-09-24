from pathlib import Path
import hashlib
import json
import math
import re
import sys


if len(sys.argv) not in (3, 4):
    raise RuntimeError(
        "Usage: token_score_check.py INPUT_FILE_OR_DIR OUT_JSON [EXPECTED]"
    )


ROOT = Path(sys.argv[1]).expanduser().resolve()
OUT = Path(sys.argv[2]).expanduser().resolve()

EXPECTED = (
    int(sys.argv[3])
    if len(sys.argv) == 4
    else None
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            x = f.read(1024 * 1024)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def role_norm(x):
    x = str(x or "").strip().lower()
    x = x.replace("_", " ")
    x = x.replace("-", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def finite_float(x):
    if isinstance(x, bool):
        return None

    try:
        v = float(x)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(v):
        return None

    return v


def get_direct_score(item, wanted):
    aliases = {
        "real": {
            "real score",
            "real_score",
        },
        "fake": {
            "fake score",
            "fake_score",
        },
    }[wanted]

    for k, v in item.items():
        if role_norm(k) in {
            role_norm(x)
            for x in aliases
        }:
            z = finite_float(v)

            if z is not None:
                return z, f"direct:{k}"

    return None, None


def get_conversation_score(item, wanted):
    convs = item.get("conversations")

    if not isinstance(convs, list):
        return None, None

    target = f"{wanted} score"

    for turn in convs:
        if not isinstance(turn, dict):
            continue

        role = None

        for rk in (
            "from",
            "role",
            "name",
        ):
            if rk in turn:
                role = role_norm(turn[rk])
                break

        if role != target:
            continue

        for vk in (
            "value",
            "content",
            "text",
            "score",
        ):
            if vk not in turn:
                continue

            z = finite_float(turn[vk])

            if z is not None:
                return z, f"conversation:{rk}:{vk}"

    return None, None


def extract_pair(item):
    if not isinstance(item, dict):
        return None

    real, real_src = get_direct_score(
        item,
        "real",
    )

    fake, fake_src = get_direct_score(
        item,
        "fake",
    )

    if real is None:
        real, real_src = get_conversation_score(
            item,
            "real",
        )

    if fake is None:
        fake, fake_src = get_conversation_score(
            item,
            "fake",
        )

    if real is None or fake is None:
        return None

    return {
        "real": real,
        "fake": fake,
        "real_source": real_src,
        "fake_source": fake_src,
    }


def candidate_lists(obj):
    out = []

    if isinstance(obj, list):
        out.append(
            ("root_list", obj)
        )

    elif isinstance(obj, dict):
        for key in (
            "images",
            "items",
            "results",
            "predictions",
        ):
            value = obj.get(key)

            if isinstance(value, list):
                out.append(
                    (key, value)
                )

    return out


if ROOT.is_file():
    files = [ROOT]

elif ROOT.is_dir():
    files = sorted(
        ROOT.rglob("*.json")
    )

else:
    raise RuntimeError(
        f"Input does not exist: {ROOT}"
    )


candidates = []

for file in files:
    try:
        obj = json.loads(
            file.read_text(
                encoding="utf-8",
                errors="strict",
            )
        )
    except Exception:
        continue

    for container_name, items in candidate_lists(obj):

        if not items:
            continue

        pairs = []

        missing = []

        for idx, item in enumerate(items):
            p = extract_pair(item)

            if p is None:
                missing.append(idx)
            else:
                pairs.append(p)

        candidates.append({
            "file": str(file),
            "container": container_name,
            "items": items,
            "pairs": pairs,
            "missing": missing,
        })


if not candidates:
    raise RuntimeError(
        f"No result-like JSON list found under {ROOT}"
    )


# Prefer the artifact with the most COMPLETE token-score pairs.
best = max(
    candidates,
    key=lambda x: (
        len(x["pairs"]),
        -len(x["missing"]),
        len(x["items"]),
    ),
)


items = best["items"]
pairs = best["pairs"]
missing = best["missing"]


# Diagnostic roles, useful if gate fails.
roles = {}

for item in items:
    if not isinstance(item, dict):
        continue

    convs = item.get("conversations")

    if not isinstance(convs, list):
        continue

    for turn in convs:
        if not isinstance(turn, dict):
            continue

        role = role_norm(
            turn.get(
                "from",
                turn.get(
                    "role",
                    turn.get(
                        "name",
                        "",
                    ),
                ),
            )
        )

        if role:
            roles[role] = (
                roles.get(role, 0)
                + 1
            )


report = {
    "input": str(ROOT),
    "files_scanned": len(files),
    "chosen_result_file": best["file"],
    "chosen_container": best["container"],
    "result_items": len(items),
    "complete_real_fake_pairs": len(pairs),
    "missing_pair_indices": missing[:50],
    "conversation_roles": roles,
    "expected": EXPECTED,
}


if pairs:
    report.update({
        "real_min": min(x["real"] for x in pairs),
        "real_max": max(x["real"] for x in pairs),
        "fake_min": min(x["fake"] for x in pairs),
        "fake_max": max(x["fake"] for x in pairs),
        "sources": sorted({
            (
                x["real_source"],
                x["fake_source"],
            )
            for x in pairs
        }),
    })


chosen = Path(
    best["file"]
)

if chosen.is_file():
    report[
        "chosen_result_sha256"
    ] = sha256(chosen)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
)


# -------------------------------------------------------------
# FAIL CLOSED.
# -------------------------------------------------------------

if len(pairs) == 0:
    raise SystemExit(40)


if len(pairs) != len(items):
    raise RuntimeError(
        "Not every inference item has BOTH numeric "
        f"REAL/FAKE token scores: "
        f"{len(pairs)}/{len(items)}"
    )


if EXPECTED is not None:
    if len(items) != EXPECTED:
        raise RuntimeError(
            f"Result coverage mismatch: "
            f"{len(items)}/{EXPECTED}"
        )


print(
    "TOKEN SCORE GATE PASS ✅ "
    f"{len(pairs)}/{len(items)}"
)
