"""Wrap images in synthetic "screenshot" frames (tweet / news article) to test gate behaviour on screenshots.

Offline, PIL only. Every source image is rendered in two frames x two sizes x two formats:
  tweet  : white card, grey round avatar, generic name + @handle, 1-2 text lines, image below, rounded corners
  article: bold headline, body line, image in the middle, caption below
  size   : the picture covers ~40% or ~80% of the canvas width
  format : PNG and JPEG q85
Usage:
    python tools/eval/make_screenshots.py --sources MANIFEST.csv --out DIR
MANIFEST.csv has columns path, set (real_screenshots | ai_screenshots), name.
Writes DIR/<set>/<name>__<frame>_<pct>.<png|jpg> and DIR/manifest.csv (path, group, label, role, source, frame, pct, fmt).
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT_DIRS = [Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts"), Path("C:/Windows/Fonts")]
CANVAS_W = 900


def font(size, bold=False):
    names = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
    for d in FONT_DIRS:
        for n in names:
            if (d / n).exists():
                return ImageFont.truetype(str(d / n), size)
    return ImageFont.load_default()


def fit_width(img, w):
    return img.resize((w, max(1, round(img.height * w / img.width))), Image.LANCZOS)


def rounded(img, radius):
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
    out = Image.new("RGB", img.size, (255, 255, 255))
    out.paste(img, mask=mask)
    return out


def tweet(img, pct):
    pic_w = int(CANVAS_W * pct / 100)
    pic = rounded(fit_width(img, pic_w), 18)
    pic = pic.crop((0, 0, pic.width, min(pic.height, 900)))
    pad = 24
    head_h = 190
    canvas = Image.new("RGB", (CANVAS_W, head_h + pic.height + 110), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.ellipse([pad, pad, pad + 64, pad + 64], fill=(190, 195, 200))
    d.text((pad + 80, pad + 2), "Người dùng A", font=font(26, True), fill=(15, 20, 25))
    d.text((pad + 80, pad + 36), "@nguoidung_a  ·  2 giờ", font=font(20), fill=(100, 110, 120))
    d.text((pad, pad + 90), "Ảnh này đang được chia sẻ rất nhiều hôm nay.", font=font(24), fill=(15, 20, 25))
    d.text((pad, pad + 124), "Mọi người nghĩ sao về bức ảnh này?", font=font(24), fill=(15, 20, 25))
    x = pad + (CANVAS_W - 2 * pad - pic.width) // 2 if pic.width < CANVAS_W - 2 * pad else pad
    canvas.paste(pic, (max(x, pad), head_h))
    y = head_h + pic.height + 28
    d.text((pad, y), "💬 128      🔁 342      ♡ 1,2K      ⇪", font=font(20), fill=(100, 110, 120))
    d.rounded_rectangle([1, 1, CANVAS_W - 2, canvas.height - 2], 14, outline=(207, 217, 222), width=2)
    return canvas


def article(img, pct):
    pic_w = int(CANVAS_W * pct / 100)
    pic = fit_width(img, pic_w)
    pic = pic.crop((0, 0, pic.width, min(pic.height, 900)))
    pad = 36
    head_h = 190
    canvas = Image.new("RGB", (CANVAS_W, head_h + pic.height + 170), (250, 250, 247))
    d = ImageDraw.Draw(canvas)
    d.text((pad, pad), "Bức ảnh gây chú ý trên mạng xã hội", font=font(34, True), fill=(20, 20, 20))
    d.text((pad, pad + 56), "Nhiều người dùng chia sẻ hình ảnh này trong ngày hôm qua.", font=font(22), fill=(60, 60, 60))
    d.text((pad, pad + 90), "Tin tức  |  Cập nhật 10:30", font=font(18), fill=(130, 130, 130))
    canvas.paste(pic, ((CANVAS_W - pic.width) // 2, head_h))
    d.text((pad, head_h + pic.height + 14), "Ảnh: nguồn minh họa. Hình ảnh chỉ mang tính tham khảo.", font=font(18), fill=(110, 110, 110))
    d.line([pad, head_h + pic.height + 60, CANVAS_W - pad, head_h + pic.height + 60], fill=(210, 210, 210), width=2)
    d.text((pad, head_h + pic.height + 78), "Nội dung bài viết tiếp tục bên dưới...", font=font(20), fill=(60, 60, 60))
    return canvas


FRAMES = {"tweet": tweet, "article": article}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for s in csv.DictReader(open(args.sources)):
        img = ImageOps.exif_transpose(Image.open(s["path"])).convert("RGB")
        label = 1 if s["set"] == "ai_screenshots" else 0
        d = args.out / s["set"]
        d.mkdir(parents=True, exist_ok=True)
        for frame, fn in FRAMES.items():
            for pct in (40, 80):
                canvas = fn(img, pct)
                for fmt, ext, kw in (("png", "png", {}), ("jpeg", "jpg", {"quality": 85})):
                    p = d / f"{s['name']}__{frame}_{pct}.{ext}"
                    canvas.save(p, fmt.upper(), **kw)
                    rows.append({"path": str(p), "group": s["set"], "label": label, "role": "select",
                                 "source": s["path"], "frame": frame, "pct": pct, "fmt": fmt})
    with open(args.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} screenshots from {len({r['source'] for r in rows})} sources -> {args.out}")


if __name__ == "__main__":
    main()
