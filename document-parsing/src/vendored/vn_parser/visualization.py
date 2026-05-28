"""Layout visualization utilities.

Filled translucent rectangles colored per category + reading-order numbers
drawn outside each block (top-right corner, red).

Color palette ported from the vn_parser draw_bbox utility.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category color (RGB, 0-255). Translucent fill rendered at alpha=0.3.
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "text":          (153, 0, 76),    # body text, references, footnotes, etc.
    "title":         (102, 102, 255), # doc_title, paragraph_title
    "figure_title":  (102, 178, 255), # captions
    "image":         (153, 255, 51),  # image / chart / header_image / footer_image
    "image_caption": (102, 178, 255),
    "image_footnote":(255, 178, 102),
    "table":         (204, 204, 0),
    "table_caption": (255, 255, 102),
    "table_footnote":(229, 255, 204),
    "formula":       (0, 255, 0),     # display_formula / inline_formula / formula_number
    "list":          (40, 169, 92),
    "seal":          (40, 169, 92),
    "discarded":     (158, 158, 158),
    "other":         (153, 0, 76),
}

# PP-DocLayoutV2 label -> category bucket above.
LABEL_TO_CATEGORY: Dict[str, str] = {
    "abstract": "text",
    "algorithm": "text",
    "aside_text": "text",
    "chart": "image",
    "content": "text",
    "display_formula": "formula",
    "doc_title": "title",
    "figure_title": "figure_title",
    "footer": "discarded",
    "footer_image": "discarded",
    "footnote": "text",
    "formula_number": "formula",
    "header": "discarded",
    "header_image": "discarded",
    "image": "image",
    "inline_formula": "formula",
    "number": "discarded",
    "paragraph_title": "title",
    "reference": "list",
    "reference_content": "text",
    "seal": "seal",
    "table": "table",
    "text": "text",
    "vertical_text": "text",
    "vision_footnote": "image_footnote",
}


def _color_for(label: str) -> Tuple[int, int, int]:
    cat = LABEL_TO_CATEGORY.get(label, "other")
    return CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort: try a real TTF, fall back to PIL default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_layout(
    image: Union[Image.Image, np.ndarray],
    blocks: Sequence[Dict],
    fill_alpha: float = 0.30,
    show_label: bool = True,
    number_color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Render the layout overlay.

    blocks: list of dicts {label, bbox=[x0,y0,x1,y1], index, score}.
    Returns a new PIL.Image (RGB).
    """
    if isinstance(image, np.ndarray):
        base = Image.fromarray(image).convert("RGBA")
    else:
        base = image.convert("RGBA")

    # Filled translucent rects on a separate layer, then alpha-composite.
    fill_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(fill_layer)
    alpha = int(round(fill_alpha * 255))
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        c = _color_for(b["label"])
        draw_fill.rectangle([x0, y0, x1, y1], fill=(c[0], c[1], c[2], alpha))
    composed = Image.alpha_composite(base, fill_layer)

    # Outlines + numbers + labels on top.
    img = composed.convert("RGB")
    draw = ImageDraw.Draw(img)
    font_idx = _font(size=max(14, int(min(img.size) / 90)))
    font_lbl = _font(size=max(10, int(min(img.size) / 130)))

    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        c = _color_for(b["label"])

        # Outline (slightly darker than fill)
        outline = (max(0, c[0] - 30), max(0, c[1] - 30), max(0, c[2] - 30))
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=2)

        # Reading-order number near top-right (outside the box).
        idx_text = str(b.get("index", ""))
        if idx_text:
            tx = x1 + 4
            ty = y0
            try:
                tw, th = draw.textbbox((0, 0), idx_text, font=font_idx)[2:]
            except Exception:
                tw, th = 14, 14
            # If number would clip the right edge, pull it inside.
            if tx + tw > img.size[0] - 2:
                tx = max(0, x1 - tw - 4)
            # Light halo for legibility.
            for dx in (-1, 1):
                for dy in (-1, 1):
                    draw.text((tx + dx, ty + dy), idx_text, fill=(255, 255, 255), font=font_idx)
            draw.text((tx, ty), idx_text, fill=number_color, font=font_idx)

        if show_label:
            label_text = b["label"]
            score = b.get("score")
            if score is not None:
                label_text = f"{label_text} {float(score):.2f}"
            try:
                tw, th = draw.textbbox((0, 0), label_text, font=font_lbl)[2:]
            except Exception:
                tw, th = 80, 12
            # Place label inside the top-left corner with a filled bg strip.
            lx, ly = x0 + 1, y0 + 1
            if ly + th > y1:
                ly = max(0, y0 - th - 2)
            draw.rectangle([lx - 1, ly - 1, lx + tw + 2, ly + th + 1], fill=outline)
            draw.text((lx, ly), label_text, fill=(255, 255, 255), font=font_lbl)

    return img


def draw_layout_array(image, blocks, **kwargs) -> np.ndarray:
    """Same as draw_layout but returns numpy RGB."""
    return np.asarray(draw_layout(image, blocks, **kwargs))
