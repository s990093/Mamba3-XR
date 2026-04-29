#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TIMESTAMP_RE = re.compile(
    r"Screenshot (\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})\s*([AP]M)",
    re.IGNORECASE,
)


def parse_timestamp(path: Path) -> datetime:
    match = TIMESTAMP_RE.search(path.stem)
    if not match:
        raise ValueError(f"Unrecognized screenshot filename: {path.name}")
    date_part, hour, minute, second, meridiem = match.groups()
    stamp = f"{date_part} {hour}:{minute}:{second} {meridiem.upper()}"
    return datetime.strptime(stamp, "%Y-%m-%d %I:%M:%S %p")


def discover_images(root: Path) -> list[tuple[datetime, Path]]:
    items = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith("Screenshot "):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".heic"}:
            continue
        try:
            items.append((parse_timestamp(path), path))
        except ValueError:
            continue
    return sorted(items, key=lambda item: item[0])


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ] + candidates
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((max_width, max_height))
    return copy


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, font, fill, page_width: int) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (page_width - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, font=font, fill=fill)
    return bbox[3] - bbox[1]


def build_cover(items: list[tuple[datetime, Path]], page_size: tuple[int, int], source_root: Path) -> Image.Image:
    width, height = page_size
    page = Image.new("RGB", page_size, "#F6F3EE")
    draw = ImageDraw.Draw(page)
    title_font = load_font(44, bold=True)
    body_font = load_font(24)
    small_font = load_font(18)

    y = 160
    y += draw_centered(draw, "Desktop Screenshot Timeline Report", y, title_font, "#1F2937", width) + 24
    y += draw_centered(draw, "依照圖片時間排序的桌面截圖報告", y, body_font, "#374151", width) + 48

    first_dt = items[0][0]
    last_dt = items[-1][0]
    summary = [
        f"來源資料夾: {source_root}",
        f"圖片總數: {len(items)}",
        f"時間範圍: {first_dt.strftime('%Y-%m-%d %H:%M:%S')} 至 {last_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"建立時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    for line in summary:
        y += draw_centered(draw, line, y, small_font, "#4B5563", width) + 18

    box = [110, y + 30, width - 110, y + 170]
    draw.rounded_rectangle(box, radius=24, fill="#E7DED1")
    note = (
        "本報告僅收錄指定資料夾中的 Screenshot 圖片，"
        "並依檔名中的時間自動排序，以下每一頁對應一張圖片。"
    )
    note_font = load_font(20)
    lines = wrap_text(note, note_font, box[2] - box[0] - 50)
    text_y = box[1] + 28
    for line in lines:
        draw.text((box[0] + 25, text_y), line, font=note_font, fill="#3F3A33")
        text_y += 30

    return page


def wrap_text(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_content_pages(items: list[tuple[datetime, Path]], page_size: tuple[int, int], source_root: Path) -> list[Image.Image]:
    width, height = page_size
    title_font = load_font(28, bold=True)
    caption_font = load_font(18)
    meta_font = load_font(20)
    pages = []

    total_pages = len(items)
    for page_index, (dt, path) in enumerate(items, start=1):
        page = Image.new("RGB", page_size, "#F7F7F5")
        draw = ImageDraw.Draw(page)
        draw.text((50, 36), dt.strftime("%Y-%m-%d %H:%M:%S"), font=title_font, fill="#111827")
        draw.text((width - 170, 42), f"{page_index}/{total_pages}", font=caption_font, fill="#6B7280")

        panel = [45, 95, width - 45, height - 100]
        draw.rounded_rectangle(panel, radius=24, fill="white", outline="#D1D5DB", width=2)

        with Image.open(path) as raw:
            fitted = fit_image(raw, width - 140, height - 260)
        img_x = (width - fitted.width) // 2
        img_y = 125 + ((height - 260) - fitted.height) // 2
        page.paste(fitted, (img_x, img_y))

        rel_path = str(path.relative_to(source_root))
        meta_text = f"{rel_path}  |  {path.name}"
        meta_lines = wrap_text(meta_text, meta_font, width - 120)
        text_y = height - 72
        for line in meta_lines[:2]:
            draw.text((60, text_y), line, font=meta_font, fill="#6B7280")
            text_y += 24
        pages.append(page)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--no-cover", action="store_true")
    args = parser.parse_args()

    items = discover_images(args.root)
    if not items:
        raise SystemExit("No matching screenshots found.")
    if args.descending:
        items = list(reversed(items))

    page_size = (2339, 1654)
    pages = build_content_pages(items, page_size, args.root)
    if not args.no_cover:
        pages = [build_cover(items, page_size, args.root), *pages]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(args.output, save_all=True, append_images=pages[1:], resolution=200.0)
    print(args.output)


if __name__ == "__main__":
    main()
