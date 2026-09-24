
from pathlib import Path
import copy
import sys
import yaml


src = Path(sys.argv[1])
dst = Path(sys.argv[2])
work = sys.argv[3]

inputs = sys.argv[4:]


cfg = yaml.safe_load(
    src.read_text(
        errors="ignore"
    )
)


def rewrite_strings(x):

    if isinstance(x, dict):

        return {
            k: rewrite_strings(v)
            for k, v in x.items()
        }

    if isinstance(x, list):

        return [
            rewrite_strings(v)
            for v in x
        ]

    if isinstance(x, str):

        return x.replace(
            "/home/aiotlab/hoangan/outputs/"
            "x2dfd_router_integration/full_DFDCP_DFDC",
            work,
        ).replace(
            "outputs/x2dfd_router_integration/"
            "full_DFDCP_DFDC",
            work,
        )

    return x


cfg = rewrite_strings(cfg)


candidates = []


def walk(x, path=()):

    if isinstance(x, dict):

        for k, v in x.items():

            if (
                isinstance(v, list)
                and v
                and all(
                    isinstance(z, str)
                    and z.lower().endswith(".json")
                    for z in v
                )
            ):

                candidates.append(
                    path + (k,)
                )

            walk(
                v,
                path + (k,),
            )

    elif isinstance(x, list):

        for i, v in enumerate(x):
            walk(
                v,
                path + (i,),
            )


walk(cfg)


if len(candidates) != 1:
    raise RuntimeError(
        "Expected exactly one JSON input list in config; "
        f"found {candidates}"
    )


target = candidates[0]


cur = cfg

for key in target[:-1]:
    cur = cur[key]


cur[target[-1]] = inputs


dst.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    )
)


print(
    "PATCHED CONFIG =",
    dst,
)

print(
    "INPUT LIST PATH =",
    target,
)

print(
    "INPUTS =",
    inputs,
)

