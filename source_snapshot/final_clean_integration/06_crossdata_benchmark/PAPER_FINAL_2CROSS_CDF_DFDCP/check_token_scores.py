#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])

def number(x):
    if isinstance(x, bool):
        return None
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None

def records(x):
    if isinstance(x, list):
        for v in x:
            yield from records(v)
    elif isinstance(x, dict):
        if isinstance(x.get("conversation"), list):
            yield x
        for key in ("records", "data", "items", "results", "conversations"):
            value = x.get(key)
            if isinstance(value, (list, dict)):
                yield from records(value)

def extract(record):
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return None

    real = []
    fake = []

    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        source = str(
            turn.get("from", turn.get("role", ""))
        ).strip().lower()

        if source not in {"real score", "fake score"}:
            continue

        value = turn.get("value", turn.get("content"))
        score = number(value)

        if score is None:
            raise ValueError(
                f"non-numeric {source}: {value!r}"
            )

        if source == "real score":
            real.append(score)
        else:
            fake.append(score)

    if len(real) != 1 or len(fake) != 1:
        raise ValueError(
            f"expected one real/fake score; "
            f"real={len(real)} fake={len(fake)}"
        )

    real_score, fake_score = real[0], fake[0]

    if not (
        0.0 <= real_score <= 1.0
        and 0.0 <= fake_score <= 1.0
    ):
        raise ValueError("score outside [0,1]")

    if abs(real_score + fake_score - 1.0) > 1e-6:
        raise ValueError("real_score + fake_score != 1")

    return real_score, fake_score

files = sorted(root.rglob("*.json"))
all_records = []

for path in files:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    all_records.extend(records(obj))

pairs = []

for index, record in enumerate(all_records):
    try:
        pair = extract(record)
    except Exception as exc:
        raise SystemExit(
            f"BAD_SCORE_RECORD index={index}: {exc}"
        )
    if pair is not None:
        pairs.append(pair)

if not pairs:
    raise SystemExit("NO_CONTINUOUS_REAL_FAKE_SCORE_PAIRS")

payload = {
    "status": "PASS",
    "files_scanned": len(files),
    "records_scanned": len(all_records),
    "numeric_real_fake_score_values": 2 * len(pairs),
    "pairs": len(pairs),
    "sample": [
        {"real_score": r, "fake_score": f}
        for r, f in pairs[:2]
    ],
}

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2))

