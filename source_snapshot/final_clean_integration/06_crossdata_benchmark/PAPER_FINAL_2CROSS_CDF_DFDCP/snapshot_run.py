
from pathlib import Path
import json
import shutil
import sys


pointer = Path(
    sys.argv[1]
)

outdir = Path(
    sys.argv[2]
)

if outdir.exists():
    shutil.rmtree(outdir)

# CANONICAL_SNAPSHOT_CLEAN
outdir.mkdir(
    parents=True,
    exist_ok=True,
)


obj = json.loads(
    pointer.read_text(
        errors="ignore"
    )
)


(outdir / "latest_run.json").write_text(
    json.dumps(
        obj,
        indent=2,
    )
)


def strings(x):

    if isinstance(x, dict):

        for v in x.values():
            yield from strings(v)

    elif isinstance(x, list):

        for v in x:
            yield from strings(v)

    elif isinstance(x, str):

        yield x


existing = []


for s in strings(obj):

    p = Path(s)

    if p.exists():
        existing.append(
            str(
                p.resolve()
            )
        )


existing = list(
    dict.fromkeys(existing)
)


(outdir / "referenced_paths.json").write_text(
    json.dumps(
        existing,
        indent=2,
    )
)


artifact_dir = (
    outdir
    / "artifacts"
)

artifact_dir.mkdir(
    exist_ok=True
)


copied = []


def maybe_copy(p):

    allowed = {
        ".json",
        ".jsonl",
        ".csv",
        ".txt",
    }

    if (
        p.is_file()
        and p.suffix.lower() in allowed
    ):

        try:
            if p.stat().st_size > 2 * 1024**3:
                return
        except:
            return

        name = (
            str(p.resolve())
            .replace("/", "__")
        )

        target = (
            artifact_dir
            / name
        )

        shutil.copy2(
            p,
            target,
        )

        copied.append(
            {
                "source": str(p),
                "copy": str(target),
            }
        )

    elif p.is_dir():

        for q in p.rglob("*"):

            if not q.is_file():
                continue

            if q.suffix.lower() not in allowed:
                continue

            try:
                if q.stat().st_size > 2 * 1024**3:
                    continue
            except:
                continue

            name = (
                str(q.resolve())
                .replace("/", "__")
            )

            target = (
                artifact_dir
                / name
            )

            shutil.copy2(
                q,
                target,
            )

            copied.append(
                {
                    "source": str(q),
                    "copy": str(target),
                }
            )


for s in existing:
    maybe_copy(
        Path(s)
    )


(outdir / "COPIED_ARTIFACTS.json").write_text(
    json.dumps(
        copied,
        indent=2,
    )
)


print(
    "REFERENCED =",
    len(existing),
)

print(
    "COPIED =",
    len(copied),
)

