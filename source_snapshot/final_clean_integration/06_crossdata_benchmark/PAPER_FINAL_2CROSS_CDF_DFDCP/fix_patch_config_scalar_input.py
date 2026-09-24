from pathlib import Path
import sys

p = Path(
    sys.argv[1]
)

if not p.is_file():
    raise RuntimeError(
        f"patch_config missing: {p}"
    )

text = p.read_text(
    encoding="utf-8",
)

compile(
    text,
    str(p),
    "exec",
)

if (
    "x2dfd-infer.inputs.images-2cross-v2"
    not in text
):
    raise RuntimeError(
        "Non-canonical patch_config detected"
    )

if (
    "Expected exactly one JSON input list in config"
    in text
):
    raise RuntimeError(
        "Legacy patcher returned"
    )

print(
    "✅ canonical patch_config verified; no mutation"
)
