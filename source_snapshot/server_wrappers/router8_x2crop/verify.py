import csv
from pathlib import Path


BASE = Path(
    "/home/aiotlab/hoangan/"
    "outputs/expert_profiling/full"
)

datasets = {
    "GANGen":
        BASE / "GANGen",

    "FFPP_X2DFD32":
        BASE / "FFPP_X2DFD32",
}

experts = [
    "blending",
    "diffusion",
    "frequency",
    "texture",
]


done = 0

print()
print("=" * 78)
print("FINAL 8 SCORE CSV STATUS")
print("=" * 78)


for dataset, folder \
in datasets.items():

    manifest = (
        folder
        / "manifest.csv"
    )

    if not manifest.exists():

        print(
            dataset,
            ": manifest missing"
        )

        continue


    with manifest.open() as f:

        rows = list(
            csv.DictReader(f)
        )


    n = len(rows)

    real = sum(
        int(r["label"]) == 0
        for r in rows
    )

    fake = sum(
        int(r["label"]) == 1
        for r in rows
    )


    print()
    print(dataset)

    print(
        f"images={n} "
        f"real={real} "
        f"fake={fake}"
    )


    for expert in experts:

        path = (
            folder
            / f"{expert}_scores.csv"
        )

        scored = set()

        if path.exists():

            with path.open() as f:

                for r in csv.DictReader(f):

                    if r.get(
                        "score",
                        ""
                    ) != "":

                        scored.add(
                            r["image_path"]
                        )


        ok = (
            n > 0
            and len(scored) == n
        )

        done += int(ok)

        print(
            f"  {expert.upper():12s}"
            f"{len(scored):8d}/{n}"
            f" {'OK' if ok else 'MISSING'}"
        )


print()
print("=" * 78)
print(
    "TOTAL SCORE CSV =",
    done,
    "/ 8"
)

if done == 8:

    print(
        "ROUTER PROFILING READY"
    )

print("=" * 78)
