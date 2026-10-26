"""Fixed regression set for the web pipeline (gate + face crop + X2DFD decision).

Layer A (every manifest image): face detector and AI-gate score from the web classes
(FaceCropper, AIImageDetector) -> face rate, gate FPR/recall per group.
Layer B (subset): final verdict = gate >= AI_GATE_THRESHOLD ? FAKE : X2DFD >= X2DFD threshold.
X2DFD needs LLaVA, which only the running web holds, so its fake_prob comes from cached API
runs (valid only when the crop is unchanged, i.e. strict Haar) or, for must-pass rows, a live call.

Usage:
    python tools/eval/regression.py --out ~/hoangan/eval_runs/<date>/baseline_before.csv [--api]
Thresholds are read from web settings, so env overrides (AI_GATE_THRESHOLD, ...) apply.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image, ImageOps  # noqa: E402

from web.backend.app.config import settings  # noqa: E402
from web.backend.app.services.face_cropper import FaceCropper  # noqa: E402
from web.model_worker.ai_image_detector import AIImageDetector  # noqa: E402

RUN = Path.home() / "hoangan/eval_runs/20261026"
MANIFEST = RUN / "large/manifest.csv"
# Cached X2DFD fake_prob from API runs on the strict-Haar pipeline: (csv, path column, prob column)
X2DFD_CACHES = [
    (RUN / "large/x2dfd_api.csv", "path", "x2dfd_fake_probability"),
    (RUN / "results.csv", "path", "fake_probability"),
]
REAL_GROUPS = ("celebdf_real", "ffhq_real", "diffusion_pilot_real")
FAKE_GROUPS = ("celebdf_fake", "ai_set")


def must_pass_rows():
    girl = Path.home() / "hoangan/01_AI_BE_GAI_OM_CHO_CON.png"
    girl_jpg = RUN / "mustpass/girl_q90.jpg"
    if not girl_jpg.exists():
        girl_jpg.parent.mkdir(parents=True, exist_ok=True)
        Image.open(girl).convert("RGB").save(girl_jpg, "JPEG", quality=90)
    rows = [
        {"path": str(girl), "group": "mustpass", "expect": "FAKE", "note": "girl PNG"},
        {"path": str(girl_jpg), "group": "mustpass", "expect": "FAKE", "note": "girl JPG q90 (derived from PNG)"},
    ]
    zalo = sorted((Path.home() / "hoangan/test_imgs").glob("z5902186781777_*.jpg"))
    rows.append({"path": str(zalo[0]) if zalo else "", "group": "mustpass", "expect": "FAKE",
                 "note": "girl Zalo" if zalo else "girl Zalo: FILE MISSING on server"})
    for p in sorted((Path.home() / "hoangan/test_imgs").glob("2aObo*")):
        rows.append({"path": str(p), "group": "mustpass", "expect": "REAL+FACE", "note": "phone photo (real)"})
    return rows


def detect_face(img):
    if hasattr(FaceCropper, "detect_face_with_fallback"):
        found = FaceCropper.detect_face_with_fallback(img)
        return found[1] if found else ""
    return "haar" if FaceCropper.detect_largest_face(img) else ""


def load_cache():
    cache = {}
    for path, pcol, vcol in X2DFD_CACHES:
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            v = r.get(vcol, "")
            if v not in ("", None):
                cache[os.path.realpath(r[pcol])] = float(v)
    return cache


def call_api(url, path):
    import httpx
    with open(path, "rb") as fh, httpx.Client(timeout=300) as c:
        resp = c.post(f"{url}/api/v1/analyze/image", files={"file": (Path(path).name, fh)})
    body = resp.json()
    if resp.status_code != 200:
        return {"api_verdict": body.get("error", {}).get("code", "HTTP_ERROR")}
    a = body.get("ai_detector") or {}
    x2 = a.get("x2dfd_fake_probability") if a.get("overridden") else body["final"]["fake_probability"]
    return {"api_verdict": body["final"]["verdict"], "api_face_crop": body.get("provenance", {}).get("face_crop", ""),
            "x2dfd": x2 if x2 is not None else float("nan")}


def decide(r, gate_thr, x_thr):
    if r["gate"] >= gate_thr:
        return "FAKE"
    if not r["face"]:
        return "NO_FACE"
    if math.isnan(r["x2dfd"]):
        return "?"
    return "FAKE" if r["x2dfd"] >= x_thr else "REAL"


def rate(rows, pred):
    return (sum(pred(r) for r in rows) / len(rows), len(rows)) if rows else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--api", action="store_true", help="call the live API for must-pass rows")
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    args = ap.parse_args()

    gate_thr = float(settings.AI_GATE_THRESHOLD)
    x_thr = float(getattr(settings, "X2DFD_DECISION_THRESHOLD", 0.5))
    det = AIImageDetector()
    cache = load_cache()

    rows = [dict(r) for r in csv.DictReader(open(MANIFEST)) if r["role"] == "select"] + must_pass_rows()
    for r in rows:
        r.setdefault("expect", ""); r.setdefault("note", "")
        if not r["path"]:
            r.update(face="", gate=float("nan"), x2dfd=float("nan"), final="MISSING")
            continue
        img = ImageOps.exif_transpose(Image.open(r["path"])).convert("RGB")
        r["face"] = detect_face(img)
        r["gate"] = det.score(Path(r["path"]))
        # Cached X2DFD is only valid for the unchanged strict-Haar crop.
        r["x2dfd"] = cache.get(os.path.realpath(r["path"]), float("nan")) if r["face"] == "haar" else float("nan")
        if args.api and r["group"] == "mustpass":
            r.update(call_api(args.url, r["path"]))
        r["final"] = decide(r, gate_thr, x_thr)

    fields = ["path", "group", "label", "expect", "note", "face", "gate", "x2dfd", "final", "api_verdict", "api_face_crop"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)

    G = lambda g: [r for r in rows if r["group"] == g]
    summary = {"gate_threshold": gate_thr, "x2dfd_threshold": x_thr, "layer_A": {}, "layer_B": {}, "mustpass": []}
    reals = [r for g in REAL_GROUPS for r in G(g)]
    print(f"gate_thr={gate_thr} x2dfd_thr={x_thr}\n\nLayer A (all images): face rate, gate FAKE rate")
    for g in REAL_GROUPS + FAKE_GROUPS + ("ALL_REAL",):
        rs = reals if g == "ALL_REAL" else G(g)
        face, n = rate(rs, lambda r: bool(r["face"]))
        gate, _ = rate(rs, lambda r: r["gate"] >= gate_thr)
        kind = "FPR" if g in REAL_GROUPS or g == "ALL_REAL" else "recall"
        summary["layer_A"][g] = {"n": n, "face_rate": face, f"gate_{kind}": gate}
        print(f"  {g:<22} n={n:<4} face={face:.3f}  gate_{kind}={gate:.3f}")

    print("\nLayer B (final verdict on images with a cached X2DFD score; unbiased subset)")
    # Only rows whose X2DFD score is known: otherwise gate-decided rows would be over-represented.
    known = lambda rs: [r for r in rs if not math.isnan(r["x2dfd"])]
    for g in REAL_GROUPS + FAKE_GROUPS + ("ALL_REAL",):
        rs = known(reals if g == "ALL_REAL" else G(g))
        v, n = rate(rs, lambda r: r["final"] == "FAKE")
        kind = "FPR" if g in REAL_GROUPS or g == "ALL_REAL" else "recall"
        summary["layer_B"][g] = {"n": n, kind: v}
        print(f"  {g:<22} n={n:<4} {kind}={v:.3f}")

    print("\nMust-pass")
    for r in G("mustpass"):
        got = r.get("api_verdict") or r["final"]
        ok = (got == "FAKE") if r["expect"] == "FAKE" else (got == "REAL" and bool(r["face"]))
        if r["final"] == "MISSING":
            ok = None
        summary["mustpass"].append({"note": r["note"], "file": Path(r["path"]).name, "expect": r["expect"], "face": r["face"],
                                    "gate": r["gate"], "x2dfd": r["x2dfd"], "sim": r["final"], "api": r.get("api_verdict", ""), "pass": ok})
        print(f"  {r['note']:<34} expect={r['expect']:<9} face={r['face'] or '-':<8} gate={r['gate']:.4f} "
              f"x2dfd={r['x2dfd']:.4f} sim={r['final']:<8} api={r.get('api_verdict', ''):<17} pass={ok}")
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
