import os
import sys
import csv
import logging
import importlib.util
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path("/home/aiotlab/hoangan")

FFROOT = (
    BASE
    / "datasets"
    / "FaceForensicsC23"
    / "FaceForensics++_C23"
)

DBROOT = BASE / "projects" / "DeepfakeBench"
PREP   = DBROOT / "preprocessing"

OUT = (
    BASE
    / "outputs"
    / "expert_profiling"
    / "full"
    / "FFPP_X2DFD32"
)

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("FF++ PREPROCESSING - X2DFD / DEEPFAKEBENCH PROTOCOL")
print("=" * 76)

print("FFROOT :", FFROOT)
print("DBROOT :", DBROOT)
print("OUT    :", OUT)

if not FFROOT.exists():
    raise RuntimeError(f"FF++ root not found: {FFROOT}")

if not (PREP / "preprocess.py").exists():
    raise RuntimeError(
        f"DeepfakeBench preprocess.py not found: {PREP/'preprocess.py'}"
    )


# ============================================================
# IMPORT OFFICIAL PREPROCESS MODULE
# ============================================================

os.chdir(PREP)

spec = importlib.util.spec_from_file_location(
    "deepfakebench_preprocess",
    PREP / "preprocess.py"
)

dbp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbp)


# ============================================================
# LOGGER REQUIRED BY OFFICIAL FUNCTIONS
# ============================================================

logger = logging.getLogger("x2dfd_ffpp_crop")
logger.setLevel(logging.INFO)

if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(sh)

dbp.logger = logger


# ============================================================
# ONLY THE 5 FF++ GROUPS USED HERE
# ============================================================

groups = [
    (
        "real",
        0,
        FFROOT / "original_sequences" / "youtube" / "c23",
    ),
    (
        "Deepfakes",
        1,
        FFROOT / "manipulated_sequences" / "Deepfakes" / "c23",
    ),
    (
        "Face2Face",
        1,
        FFROOT / "manipulated_sequences" / "Face2Face" / "c23",
    ),
    (
        "FaceSwap",
        1,
        FFROOT / "manipulated_sequences" / "FaceSwap" / "c23",
    ),
    (
        "NeuralTextures",
        1,
        FFROOT / "manipulated_sequences" / "NeuralTextures" / "c23",
    ),
]


print()
print("Video discovery:")

jobs = []

for cls, label, c23root in groups:

    video_dir = c23root / "videos"

    videos = sorted(video_dir.glob("*.mp4"))

    print(f"{cls:15s}: {len(videos)} videos")

    for video in videos:
        jobs.append(
            (cls, label, c23root, video)
        )

print("TOTAL VIDEOS:", len(jobs))

if len(jobs) == 0:
    raise RuntimeError("No FF++ videos found.")


# ============================================================
# RESUME MARKERS
#
# Official DeepfakeBench itself writes:
#   c23/frames/<video>/*.png
#   c23/landmarks/<video>/*.npy
#
# We add ONLY a marker folder so rerunning does not recrop
# completed videos.
# ============================================================

def process_one(job):

    cls, label, c23root, video = job

    marker_dir = c23root / ".x2dfd32_done"
    marker_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    marker = marker_dir / f"{video.stem}.done"

    if marker.exists():
        frame_dir = c23root / "frames" / video.stem

        n = len(
            list(frame_dir.glob("*.png"))
        ) if frame_dir.exists() else 0

        return (
            cls,
            video.stem,
            "skip",
            n
        )


    # --------------------------------------------------------
    # EXACT OFFICIAL DEEPFAKEBENCH FUNCTION
    #
    # mode='fixed_num_frames'
    # num_frames=32
    #
    # mask=None because masks are not required for expert
    # scoring; crop transform is identical.
    # --------------------------------------------------------

    dbp.video_manipulate(
        movie_path=video,
        mask_path=None,
        dataset_path=c23root,
        mode="fixed_num_frames",
        num_frames=32,
        stride=10,
    )

    frame_dir = (
        c23root
        / "frames"
        / video.stem
    )

    n = len(
        list(frame_dir.glob("*.png"))
    ) if frame_dir.exists() else 0

    marker.write_text(str(n))

    return (
        cls,
        video.stem,
        "done",
        n
    )


# Same output regardless of thread count.
# Keep moderate concurrency to avoid launching thousands of
# dlib predictors simultaneously.
workers = min(
    8,
    os.cpu_count() or 1
)

print()
print("Workers:", workers)
print("Starting/resuming official crop...")


completed = 0

with ThreadPoolExecutor(
    max_workers=workers
) as pool:

    futures = [
        pool.submit(
            process_one,
            job
        )
        for job in jobs
    ]

    total = len(futures)

    for future in as_completed(futures):

        completed += 1

        try:
            cls, vid, status, n = future.result()

            if (
                completed % 25 == 0
                or status == "done"
            ):
                print(
                    f"[{completed}/{total}] "
                    f"{cls:15s} "
                    f"{vid:12s} "
                    f"{status:5s} "
                    f"faces={n}"
                )

        except Exception as e:

            print(
                f"[{completed}/{total}] ERROR:",
                repr(e)
            )


# ============================================================
# BUILD MANIFEST FROM OFFICIAL CROPPED FACE FOLDERS
# ============================================================

rows = []

for cls, label, c23root in groups:

    root = c23root / "frames"

    if not root.exists():
        continue

    for video_dir in sorted(
        p for p in root.iterdir()
        if p.is_dir()
    ):

        for image in sorted(
            video_dir.glob("*.png")
        ):

            rows.append({
                "image_path":
                    str(image.resolve()),

                "label":
                    label,

                "generator":
                    cls,

                "domain":
                    "FFPP",

                "group":
                    video_dir.name,
            })


manifest = OUT / "manifest.csv"

with manifest.open(
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_path",
            "label",
            "generator",
            "domain",
            "group",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


counts = Counter(
    r["generator"]
    for r in rows
)

real = sum(
    r["label"] == 0
    for r in rows
)

fake = sum(
    r["label"] == 1
    for r in rows
)


print()
print("=" * 76)
print("X2DFD / DEEPFAKEBENCH CROPPED FF++ MANIFEST")
print("=" * 76)

for cls, _, _ in groups:
    print(
        f"{cls:15s}:",
        counts.get(cls, 0)
    )

print()
print("REAL :", real)
print("FAKE :", fake)
print("TOTAL:", len(rows))

print()
print("THEORETICAL MAX:")
print("REAL : 32000")
print("FAKE : 128000")
print("TOTAL: 160000")

print()
print("Manifest:", manifest)
print("=" * 76)
