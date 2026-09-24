from pathlib import Path
import csv
import sys

RUN = Path(sys.argv[1])
STOP = Path(sys.argv[2])

def fail(msg):
    STOP.write_text(
        "DFDC_RUNTIME_GUARD\n"
        + msg
        + "\n",
        encoding="utf-8",
    )
    print(msg)
    raise SystemExit(2)

checked = 0

for p in RUN.rglob("manifest.csv"):

    try:
        with p.open(
            newline="",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        continue

    if not rows:
        continue

    cols = set(rows[0])

    dataset_col = (
        "dataset"
        if "dataset" in cols
        else None
    )

    path_col = next(
        (
            x
            for x in (
                "image_path",
                "path",
                "image",
            )
            if x in cols
        ),
        None,
    )

    if dataset_col is None and path_col is None:
        continue

    checked += 1

    for i, row in enumerate(rows):

        if dataset_col:

            ds = str(
                row.get(
                    dataset_col,
                    "",
                )
            ).strip().upper()

            # Exact DFDC only. DFDCP is allowed.
            if ds == "DFDC":
                fail(
                    f"DFDC dataset row leaked into {p}, row={i}"
                )

        if path_col:

            s = str(
                row.get(
                    path_col,
                    "",
                )
            ).replace(
                "\\",
                "/",
            ).upper()

            if "/DFDC/" in s:
                fail(
                    f"DFDC image path leaked into {p}, row={i}: {s}"
                )

print(
    f"2-cross runtime guard PASS; manifests_checked={checked}"
)
