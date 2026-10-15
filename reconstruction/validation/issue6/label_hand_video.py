"""Add an estimate disclosure without requiring FFmpeg's optional drawtext filter."""

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def label_video(source, output):
    if output.exists():
        raise FileExistsError(output)
    font = ImageFont.load_default()
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(name, 16)
            break
        except OSError:
            pass
    label = Image.new("RGB", (1280, 28), "black")
    ImageDraw.Draw(label).text(
        (10, 4),
        "Blue left - Yellow right - Includes estimated poses",
        font=font,
        fill="white",
    )
    image = output.with_suffix(".label.png")
    label.save(image)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-loop",
            "1",
            "-i",
            str(image),
            "-filter_complex",
            "[0:v]pad=iw:ih+28:0:0:color=black[base];[base][1:v]overlay=0:H-h:shortest=1",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    label_video(args.source, args.output)
