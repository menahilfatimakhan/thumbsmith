"""Stage 3 — lay the headline onto the rendered image.

The headline is composited here rather than painted by the image model because this is the
one stage that has to be exact: the right words, spelled correctly, at a size that survives
being shrunk to 210x118, in a place that does not cover the subject's face or collide with
YouTube's own duration badge.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from . import config, extract

MAX_LINES_COLUMN = 3
MAX_LINES_BAND = 2

# A zone needs a face-free strip at least this tall (as a fraction of the image) to be
# usable — roughly two lines at the minimum font size.
MIN_CLEARANCE_FRACTION = 0.17

Box = tuple[float, float, float, float]


def _stroke_for(font_size: int) -> int:
    """Outline thickness. Scales with the type so it stays proportional at any size."""
    return max(3, font_size // 11)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _crop_box_for_aspect(w: int, h: int, target_w: int, target_h: int,
                         focus: tuple[float, float] | None = None) -> tuple[int, int, int, int]:
    """Crop to the target aspect ratio, keeping `focus` (0-1 relative point) in frame if given."""
    target_ratio = target_w / target_h
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-3:
        return (0, 0, w, h)

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        centre = focus[0] * w if focus else w / 2
        left = int(max(0, min(centre - new_w / 2, w - new_w)))
        return (left, 0, left + new_w, h)

    new_h = int(w / target_ratio)
    centre = focus[1] * h if focus else h / 2
    top = int(max(0, min(centre - new_h / 2, h - new_h)))
    return (0, top, w, top + new_h)


def _safe_rect(w: int, h: int) -> Box:
    margin_x, margin_y = w * config.SAFE_MARGIN_FRACTION, h * config.SAFE_MARGIN_FRACTION
    return (margin_x, margin_y, w - margin_x, h - margin_y)


def _badge_rect(w: int, h: int) -> Box:
    """The bottom-right corner YouTube covers with the video duration."""
    return (w * (1 - config.DURATION_BADGE_W_FRACTION), h * (1 - config.DURATION_BADGE_H_FRACTION), w, h)


def _clearance(zone: Box, face_box: Box | None) -> float:
    """Height of the tallest strip of `zone` the face does not cut through, in pixels.

    This is the honest question to ask of a zone — not "how much area does the face cover",
    which lets a small, dead-centre face look harmless in a wide top band that the headline
    will then be painted straight across. What matters is whether a contiguous strip tall
    enough for the text survives.
    """
    zone_top, zone_bottom = zone[1], zone[3]
    height = zone_bottom - zone_top

    if face_box is None:
        return height
    # The text fills its zone's width, so a face beside the zone is not in the way at all.
    if face_box[2] <= zone[0] or face_box[0] >= zone[2]:
        return height
    face_top, face_bottom = face_box[1], face_box[3]
    if face_bottom <= zone_top or face_top >= zone_bottom:
        return height

    return max(face_top - zone_top, zone_bottom - face_bottom)


def _zone_for_side(side: str, w: int, h: int) -> Box:
    """The rectangle the headline may occupy, already inside the safe area and clear of the badge."""
    left, top, right, bottom = _safe_rect(w, h)

    if side == "left":
        return (left, top, left + (right - left) * 0.46, bottom)
    if side == "right":
        # The badge lives bottom-right, so a right-hand column has to stop short of it.
        return (right - (right - left) * 0.46, top, right, h * (1 - config.DURATION_BADGE_H_FRACTION))
    if side == "top":
        return (left, top, right, top + (bottom - top) * 0.38)
    return (left, bottom - (bottom - top) * 0.40, right, h * (1 - config.DURATION_BADGE_H_FRACTION))


def _choose_zone(preferred: str, face_box: Box | None, w: int, h: int) -> tuple[str, Box]:
    """Take the director's preferred side, then verify it against where the face actually landed.

    The image model is free to reposition the subject, so the plan's idea of which half is
    empty can be wrong by the time the render comes back. Each side is scored by how much of
    the face it would cover, and the lightest wins — but the director's choice is discounted,
    because it asked for that side to be left clear and usually got it. Without the discount a
    centred subject bounces the headline to whichever side happens to be a pixel emptier,
    ignoring the plan entirely.
    """
    sides = ("left", "right", "bottom", "top")
    if preferred not in sides:
        preferred = "bottom"

    zones = {side: _zone_for_side(side, w, h) for side in sides}
    if face_box is None:
        return preferred, zones[preferred]

    room = {side: _clearance(zone, face_box) for side, zone in zones.items()}

    # Stay with the director's side whenever it can hold the text — it asked the image model
    # to leave that side clear, and second-guessing it on a technicality throws the plan away.
    if room[preferred] >= h * MIN_CLEARANCE_FRACTION:
        return preferred, zones[preferred]

    side = max(sides, key=lambda s: room[s])
    return side, zones[side]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _boost_colors(img: Image.Image, strength: float) -> Image.Image:
    """Give the frame thumbnail 'pop'. strength 1.0 = full grade for a raw video still."""
    if strength <= 0:
        return img
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1 + 0.20 * strength)
    img = ImageEnhance.Color(img).enhance(1 + 0.35 * strength)
    img = ImageEnhance.Sharpness(img).enhance(1 + 0.15 * strength)
    return img


def _scrim(img: Image.Image, side: str, strength: float) -> Image.Image:
    """Fade a dark gradient in from the headline's side so type always has something to sit on."""
    if strength <= 0.02:
        return img

    w, h = img.size
    peak = max(0.0, min(215 * strength, 215.0))
    horizontal = side in ("left", "right")
    span = w if horizontal else h

    # Solid at the headline's edge, gone by ~70% of the way across, so the subject on the
    # far side is never touched by the shade.
    t = np.linspace(0.0, 1.0, span)
    if side in ("right", "bottom"):
        t = 1.0 - t
    ramp = (peak * np.clip(1.0 - t / 0.70, 0.0, 1.0) ** 1.6).astype(np.uint8)

    mask_array = np.tile(ramp, (h, 1)) if horizontal else np.tile(ramp[:, None], (1, w))
    mask = Image.fromarray(mask_array, mode="L")
    shade = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(shade, img, mask)


def _zone_business(img: Image.Image, zone: Box) -> tuple[float, float]:
    """(mean brightness 0-1, detail 0-1) under the headline zone — drives the scrim strength."""
    crop = img.crop((int(zone[0]), int(zone[1]), int(zone[2]), int(zone[3]))).convert("L")
    if crop.width < 2 or crop.height < 2:
        return 0.5, 0.5
    stats = crop.resize((64, 36))
    pixels = list(stats.getdata())
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return mean / 255.0, min(variance ** 0.5 / 70.0, 1.0)


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in config.FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1 cannot size the default font
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    for word in text.split():
        trial = " ".join(current + [word])
        if draw.textlength(trial, font=font) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _fit(draw: ImageDraw.ImageDraw, text: str, zone: Box, max_lines: int,
         max_size: int, min_size: int) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    """Largest font size whose wrapped lines fit inside the zone. Returns (font, lines, line_height)."""
    zone_w, zone_h = zone[2] - zone[0], zone[3] - zone[1]
    size = max(max_size, min_size)
    fallback = None

    while size >= min_size:
        font = _load_font(size)
        # textlength measures the glyphs only. The outline adds _stroke_for(size) on every
        # side, and without allowing for it the type overhangs the safe margin — which is
        # exactly the edge YouTube crops on some surfaces.
        usable_w = zone_w - 2 * _stroke_for(size)
        lines = _wrap(draw, text, font, usable_w)
        line_height = int(size * 1.06)
        widest = max((draw.textlength(line, font=font) for line in lines), default=0)

        if len(lines) <= max_lines and widest <= usable_w and line_height * len(lines) <= zone_h:
            return font, lines, line_height
        if fallback is None and len(lines) <= max_lines:
            fallback = (font, lines, line_height)
        size -= 4

    if fallback:
        return fallback
    font = _load_font(min_size)
    usable_w = zone_w - 2 * _stroke_for(min_size)
    return font, _wrap(draw, text, font, usable_w)[:max_lines], int(min_size * 1.06)


def _draw_line(draw: ImageDraw.ImageDraw, xy: tuple[float, float], line: str,
               font: ImageFont.FreeTypeFont, accent_word: str, accent_color: str,
               stroke_width: int) -> None:
    """Draw one line word by word so the accent word can take a different colour."""
    x, y = xy
    space = draw.textlength(" ", font=font)
    for word in line.split():
        fill = accent_color if accent_word and word == accent_word else "white"
        draw.text((x, y), word, font=font, fill=fill,
                  stroke_width=stroke_width, stroke_fill="black")
        x += draw.textlength(word, font=font) + space


def _block_top(zone: Box, block_height: float, face_box: Box | None) -> float:
    """Vertical position for the text block inside its zone, dodging the face if it can.

    Centred in the zone by default. When the face intrudes, the block slides into whichever
    side of it has room — which is what lets a tall column still work for a subject whose
    head sits high in the frame, instead of bouncing the headline to a different side.
    """
    zone_top, zone_bottom = zone[1], zone[3]
    centred = zone_top + max(0.0, (zone_bottom - zone_top - block_height) / 2)

    if face_box is None:
        return centred

    face_top, face_bottom = face_box[1], face_box[3]
    if face_bottom <= zone_top or face_top >= zone_bottom:
        return centred  # the face is not in this zone's vertical range at all
    if centred + block_height <= face_top or centred >= face_bottom:
        return centred  # already clear of it

    pad = block_height * 0.08
    space_below = zone_bottom - (face_bottom + pad)
    space_above = (face_top - pad) - zone_top

    if space_below >= block_height and space_below >= space_above:
        return face_bottom + pad + max(0.0, (space_below - block_height) / 2)
    if space_above >= block_height:
        return zone_top + max(0.0, (space_above - block_height) / 2)
    return centred  # no room either way — the zone choice already picked the least-bad option


def _draw_headline(img: Image.Image, headline: str, accent_word: str, accent_color: str,
                   side: str, zone: Box, face_box: Box | None) -> None:
    draw = ImageDraw.Draw(img)
    max_lines = MAX_LINES_BAND if side in ("top", "bottom") else MAX_LINES_COLUMN
    font, lines, line_height = _fit(
        draw, headline, zone, max_lines,
        max_size=int(img.height * 0.23), min_size=int(img.height * 0.075),
    )
    if not lines:
        return

    stroke_width = _stroke_for(font.size)
    block_height = line_height * len(lines)
    top = _block_top(zone, block_height, face_box)

    def line_x(line: str) -> float:
        width = draw.textlength(line, font=font)
        if side == "left":
            return zone[0] + stroke_width
        if side == "right":
            return zone[2] - width - stroke_width
        return zone[0] + (zone[2] - zone[0] - width) / 2

    # Soft drop shadow first, on its own layer, so the type lifts off the picture.
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    offset = max(2, font.size // 18)
    for i, line in enumerate(lines):
        shadow_draw.text(
            (line_x(line) + offset, top + i * line_height + offset), line,
            font=font, fill=(0, 0, 0, 190), stroke_width=stroke_width, stroke_fill=(0, 0, 0, 190),
        )
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(3, font.size // 12)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB"))

    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        _draw_line(draw, (line_x(line), top + i * line_height), line,
                   font, accent_word, accent_color, stroke_width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def make_thumbnail(image_path: str, output_path: str, headline: str = "", accent_word: str = "",
                   accent_color: str = config.DEFAULT_ACCENT_COLOR, preferred_side: str = "bottom",
                   grade_strength: float = 1.0) -> dict:
    """Crop to 16:9, grade, place the headline clear of the face and the duration badge, save.

    grade_strength is 1.0 for a raw video still and near 0 for an image the render stage
    already graded — double-grading a GPT Image 2 output crushes the highlights.

    Returns a small dict describing where the headline ended up, for logging and the UI.
    """
    img = Image.open(image_path).convert("RGB")

    # Crop around the face when there is one, so a 16:9 crop of a tall frame does not
    # slice the subject's head off.
    source = cv2.imread(image_path)
    source_face = extract.detect_face_box(source) if source is not None else None
    focus = None
    if source_face is not None:
        fx, fy, fw, fh = source_face
        focus = ((fx + fw / 2) / img.width, (fy + fh / 2) / img.height)

    crop_box = _crop_box_for_aspect(img.width, img.height, config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT, focus)
    img = img.crop(crop_box).resize((config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT), Image.LANCZOS)
    img = _boost_colors(img, grade_strength)

    # Re-detect on the final canvas: the crop, the resize, and the render itself all move
    # the face, and placement has to answer to where it is *now*.
    final = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    face = extract.detect_face_box(final)
    face_box: Box | None = (face[0], face[1], face[0] + face[2], face[1] + face[3]) if face else None

    side, zone = _choose_zone(preferred_side, face_box, img.width, img.height)

    if headline.strip():
        brightness, detail = _zone_business(img, zone)
        img = _scrim(img, side, min(0.35 + 0.45 * detail + 0.25 * brightness, 1.0))
        _draw_headline(img, headline.strip().upper(), accent_word.strip().upper(),
                       accent_color, side, zone, face_box)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, "JPEG", quality=92, optimize=True)

    return {"text_side": side, "face_detected": face_box is not None}
