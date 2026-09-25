"""Encode supplied icon masters into PNG/ICO assets (requires Pillow).

Run manually when changing artwork; Pillow is not an application dependency.
The small master deliberately omits text. No artwork is redrawn here.
"""
import argparse
from io import BytesIO
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("large", type=Path, help="Transparent master with Wune text")
    parser.add_argument("small", type=Path, help="Transparent text-free master")
    args = parser.parse_args()
    from PIL import Image
    assets = Path(__file__).resolve().parents[1] / "wune" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    large = Image.open(args.large).convert("RGBA")
    small = Image.open(args.small).convert("RGBA")
    for master in (large, small):
        if master.width != master.height:
            raise ValueError("Icon masters must be square")
    large = large.resize((256, 256), Image.Resampling.LANCZOS)
    small = small.resize((64, 64), Image.Resampling.LANCZOS)
    large.save(assets / "wune.png")
    small.save(assets / "wune-window.png")
    sizes = (16, 24, 32, 48, 64, 96, 128, 256)
    directory, frames = [], []
    offset = 6 + 16 * len(sizes)
    for size in sizes:
        master = small if size <= 32 else large
        frame = BytesIO()
        master.resize((size, size), Image.Resampling.LANCZOS).save(frame, format="PNG")
        data = frame.getvalue()
        directory.append(struct.pack("<BBBBHHII", size % 256, size % 256,
                                     0, 0, 1, 32, len(data), offset))
        frames.append(data)
        offset += len(data)
    (assets / "Wune.ico").write_bytes(struct.pack("<HHH", 0, 1, len(sizes))
                                      + b"".join(directory) + b"".join(frames))


if __name__ == "__main__":
    main()
