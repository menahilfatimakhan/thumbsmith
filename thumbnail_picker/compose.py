import os

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from . import config

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
]


def _crop_box_for_aspect(w: int, h: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    target_ratio = target_w / target_h
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-3:
        return (0, 0, w, h)

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return (0, top, w, top + new_h)


def _map_face_box(face_box: tuple[int, int, int, int], crop_box: tuple[int, int, int, int], scale: float,
                   canvas_w: int, canvas_h: int) -> tuple[float, float, float, float] | None:
    """Map a face box from the original frame's pixel space into the final cropped+resized canvas."""
    x, y, w, h = face_box
    left, top, right, bottom = crop_box
    x0 = (x - left) * scale
    y0 = (y - top) * scale
    x1 = (x + w - left) * scale
    y1 = (y + h - top) * scale

    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(canvas_w), x1), min(float(canvas_h), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _boost_colors(img: Image.Image) -> Image.Image:
    """Give the frame a proper-thumbnail 'pop': punchier contrast, saturation, and sharpness."""
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Color(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        width = draw.textlength(trial, font=font)
        if width <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _fit_caption(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int, min_size: int):
    """Shrink the font until the caption wraps to at most 2 lines that fit max_width."""
    size = max(max_size, min_size)
    while size >= min_size:
        font = _load_font(size)
        lines = _wrap_text(draw, text, font, max_width)
        widest = max(draw.textlength(line, font=font) for line in lines)
        if len(lines) <= 2 and widest <= max_width:
            return font, lines
        size -= 4
    font = _load_font(min_size)
    return font, _wrap_text(draw, text, font, max_width)[:2]


def _text_band(img_w: int, img_h: int, margin: int, min_font_size: int,
               mapped_face_box: tuple[float, float, float, float] | None) -> tuple[str, float, float]:
    """Pick the vertical strip (top or bottom) to put the caption in, clearing the face if we know where it is."""
    if mapped_face_box is None:
        return "bottom", float(margin), float(img_h - margin)

    fx0, fy0, fx1, fy1 = mapped_face_box
    face_margin = img_h * 0.035

    top_space = fy0 - face_margin - margin
    bottom_space = (img_h - margin) - (fy1 + face_margin)

    if max(top_space, bottom_space) < min_font_size:
        # face fills almost the whole frame — fall back to a bottom band as a last resort
        return "bottom", float(margin), float(img_h - margin)

    if top_space >= bottom_space:
        return "top", float(margin), fy0 - face_margin
    return "bottom", fy1 + face_margin, float(img_h - margin)


def _draw_caption(img: Image.Image, caption: str, mapped_face_box: tuple[float, float, float, float] | None) -> None:
    if not caption:
        return
    caption = caption.strip().upper()
    draw = ImageDraw.Draw(img)

    margin = int(img.width * 0.05)
    max_text_width = img.width - 2 * margin
    default_max_font = int(img.height * 0.15)
    min_font_size = int(img.height * 0.06)

    position, band_top, band_bottom = _text_band(img.width, img.height, margin, min_font_size, mapped_face_box)
    band_height = max(band_bottom - band_top, min_font_size)
    max_font_size = max(min(default_max_font, int(band_height * 0.9)), min_font_size)

    font, lines = _fit_caption(draw, caption, max_text_width, max_font_size, min_font_size)
    stroke_width = max(3, font.size // 13)
    joined = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), joined, font=font, stroke_width=stroke_width, align="center", spacing=10)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (img.width - text_w) // 2 - bbox[0]
    if position == "top":
        y = band_top - bbox[1]
    else:
        y = band_bottom - text_h - bbox[1]

    # Translucent rounded panel behind the text — guarantees contrast against any background,
    # and reads as a deliberately "designed" element rather than text just slapped on the frame.
    pad_x, pad_y = int(font.size * 0.45), int(font.size * 0.3)
    panel_box = (
        x + bbox[0] - pad_x, y + bbox[1] - pad_y,
        x + bbox[0] + text_w + pad_x, y + bbox[1] + text_h + pad_y,
    )
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(panel_box, radius=int(font.size * 0.25), fill=(0, 0, 0, 110))
    composited = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.paste(composited)

    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (x, y), joined, font=font, fill="white",
        stroke_width=stroke_width, stroke_fill="black",
        align="center", spacing=10,
    )


def make_thumbnail(frame_path: str, output_path: str, caption: str = "",
                    face_box: tuple[int, int, int, int] | None = None) -> None:
    """Crop to 16:9, resize to 1280x720, boost color/contrast, overlay the caption avoiding the face."""
    img = Image.open(frame_path).convert("RGB")
    orig_w, orig_h = img.size

    crop_box = _crop_box_for_aspect(orig_w, orig_h, config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)
    cropped = img.crop(crop_box)
    scale = config.OUTPUT_WIDTH / cropped.width
    img = cropped.resize((config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT), Image.LANCZOS)
    img = _boost_colors(img)

    mapped_face_box = None
    if face_box is not None:
        mapped_face_box = _map_face_box(face_box, crop_box, scale, config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT)

    _draw_caption(img, caption, mapped_face_box)
    img.save(output_path, "JPEG", quality=90)
