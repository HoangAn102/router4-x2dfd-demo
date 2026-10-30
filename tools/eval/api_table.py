"""Send every image of a folder to the running web API and write one CSV row per image (report only).

    python tools/eval/api_table.py --images DIR --out OUT.csv [--url http://127.0.0.1:8010] [--labels labels.csv]
labels.csv (optional): file,label  (REAL/FAKE/unknown)
"""
import argparse
import csv
import json
import time
import urllib.request
import uuid
from pathlib import Path

FIELDS = ["file", "label", "status", "error_code", "face_crop", "router_expert", "router_conf", "expert_raw",
          "expert_cal", "verdict", "fake_probability", "reliability", "llava_uses_image", "seconds"]


def post(url, path):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{url}/api/v1/analyze/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--labels", type=Path)
    a = ap.parse_args()
    labels = {r["file"]: r["label"] for r in csv.DictReader(open(a.labels))} if a.labels else {}
    rows = []
    for p in sorted(x for x in a.images.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}):
        t0 = time.time()
        d = post(a.url, p)
        router, expert, final = d.get("router") or {}, d.get("expert") or {}, d.get("final") or {}
        prov, rel = d.get("provenance") or {}, d.get("reliability") or {}
        rows.append(dict(file=p.name, label=labels.get(p.name, ""), status=d.get("status"),
                         error_code=(d.get("error") or {}).get("code", ""), face_crop=prov.get("face_crop", ""),
                         router_expert=router.get("selected_expert", ""), router_conf=router.get("router_confidence", ""),
                         expert_raw=expert.get("raw_score", ""), expert_cal=expert.get("calibrated_score", ""),
                         verdict=final.get("verdict", ""), fake_probability=final.get("fake_probability", ""),
                         reliability=rel.get("level", ""), llava_uses_image=prov.get("llava_uses_image", ""),
                         seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
