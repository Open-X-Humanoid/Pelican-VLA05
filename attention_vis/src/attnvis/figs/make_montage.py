"""Per-directory frame montage — grid every frame_*.png in a {view}/ dir into
one image so you can eyeball frames quickly.

Each leaf directory containing frame_*.png gets a `_montage.png` grid
(each cell labels the frame index in the top-left). Pure PIL, no GPU/torch.

# Usage:
#   python -m attnvis fig montage                 # scan the whole dense tree
#   python -m attnvis fig montage --root <dir>    # limit to a subtree
#   python -m attnvis fig montage --cols 6        # customize columns
"""
import argparse
import glob
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from attnvis.paths import DENSE
PAD = 4            # spacing between cells
LABEL_H = 16       # height of the frame-index strip
BG = (245, 245, 245)


def _font():
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, 12)
    return ImageFont.load_default()


def _frame_no(p: str) -> str:
    m = re.search(r"frame_(\d+)\.png", p)
    return str(int(m.group(1))) if m else "?"


def montage(d: Path, cols: int, font) -> Path | None:
    fs = sorted(glob.glob(str(d / "frame_*.png")))
    if not fs:
        return None
    tiles = [Image.open(f).convert("RGB") for f in fs]
    w, h = tiles[0].size
    n = len(tiles)
    rows = (n + cols - 1) // cols
    cw, ch = w + PAD, h + LABEL_H + PAD
    canvas = Image.new("RGB", (cols * cw + PAD, rows * ch + PAD), BG)
    draw = ImageDraw.Draw(canvas)
    for i, (t, fp) in enumerate(zip(tiles, fs)):
        r, c = divmod(i, cols)
        x, y = PAD + c * cw, PAD + r * ch
        draw.text((x + 2, y), f"f{_frame_no(fp)}", fill=(20, 20, 20), font=font)
        canvas.paste(t, (x, y + LABEL_H))
    out = d / "_montage.png"
    canvas.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DENSE), help="Root directory to scan")
    ap.add_argument("--cols", type=int, default=6, help="Grid column count")
    args = ap.parse_args()
    font = _font()
    dirs = sorted({Path(p).parent
                   for p in glob.glob(f"{args.root}/**/frame_*.png", recursive=True)})
    n = 0
    for d in dirs:
        out = montage(d, args.cols, font)
        if out:
            n += 1
            print(f"[montage] {out}", flush=True)
    print(f"\ndone: wrote _montage.png in {n} directories", flush=True)


if __name__ == "__main__":
    main()
