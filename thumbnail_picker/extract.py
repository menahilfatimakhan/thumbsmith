import glob
import os
import subprocess
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Candidate:
    id: str
    path: str
    timestamp: float
    score: float
    sharpness: float
    has_face: bool
    exposure_score: float
    colorfulness: float
    composition: float
    face_fraction: float
    face_box: tuple[int, int, int, int] | None  # (x, y, w, h) in the original frame's pixel space
    phash: int  # perceptual hash, only used to spot near-duplicate frames


def _sample_interval(duration: float) -> float:
    if duration <= 0:
        return config.SAMPLE_INTERVAL_SECONDS
    max_count = config.MAX_SAMPLED_FRAMES
    natural_count = duration / config.SAMPLE_INTERVAL_SECONDS
    if natural_count <= max_count:
        return config.SAMPLE_INTERVAL_SECONDS
    return duration / max_count


def sample_frames(video_path: str, duration: float, frames_dir: str) -> list[tuple[str, float]]:
    """Extract frames at a fixed interval via ffmpeg, skipping intro/outro. Returns (path, timestamp) pairs."""
    os.makedirs(frames_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
        os.remove(stale)

    interval = _sample_interval(duration)

    skip = duration * config.SKIP_INTRO_OUTRO_FRACTION
    start = skip
    end = max(start, duration - skip)
    trimmed_duration = max(end - start, 0)

    fps = 1.0 / interval
    pattern = os.path.join(frames_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(trimmed_duration),
        "-vf", f"fps={fps}",
        "-qscale:v", "2",
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    return [(p, start + i * interval) for i, p in enumerate(frame_paths)]


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def _exposure_score(gray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_px = gray.size
    dark_frac = hist[:16].sum() / total_px
    bright_frac = hist[240:].sum() / total_px
    mean_brightness = gray.mean()
    brightness_closeness = 1.0 - abs(mean_brightness - 125) / 125
    return max(0.0, min(brightness_closeness - (dark_frac + bright_frac), 1.0))


def _colorfulness(frame) -> float:
    """Hasler-Süsstrunk colourfulness, normalised to 0-1.

    Flat, grey, washed-out frames disappear in a YouTube grid no matter how sharp they
    are. This is the cheapest reliable proxy for "does this frame have any punch".
    """
    b, g, r = cv2.split(frame.astype("float32"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(min((std_root + 0.3 * mean_root) / 110.0, 1.0))


def _face_scores(face_box: tuple[int, int, int, int] | None, width: int, height: int) -> tuple[float, float, float]:
    """Return (face_fraction, face_quality, composition) for a detected face.

    face_quality rewards a face that is actually big enough to carry a thumbnail — a
    speck of a face in a wide shot scores near zero even though "has_face" is true.
    composition rewards the subject sitting off-centre, which is what leaves usable
    negative space for the headline.
    """
    if face_box is None:
        return 0.0, 0.0, 0.35  # no face: neutral-ish composition, we cannot tell

    x, y, w, h = face_box
    fraction = (w * h) / float(width * height)

    if fraction < config.MIN_USEFUL_FACE_AREA_FRACTION:
        quality = 0.0
    elif fraction >= config.IDEAL_FACE_AREA_FRACTION:
        # Past the ideal, bigger is not better — an extreme close-up crops out context
        # and leaves nowhere for text, so taper back down gently.
        quality = max(0.55, 1.0 - (fraction - config.IDEAL_FACE_AREA_FRACTION) * 1.2)
    else:
        quality = fraction / config.IDEAL_FACE_AREA_FRACTION

    centre_x = (x + w / 2) / width
    centre_y = (y + h / 2) / height
    # Rule of thirds horizontally: peak at 1/3 or 2/3, worst dead-centre or at the edges.
    horizontal = 1.0 - min(abs(centre_x - 1 / 3), abs(centre_x - 2 / 3)) * 3.0
    # Vertically, faces want to sit in the upper-middle, not the floor of the frame.
    vertical = 1.0 - abs(centre_y - 0.42) * 2.0
    composition = max(0.0, min(0.65 * horizontal + 0.35 * vertical, 1.0))

    return float(fraction), float(quality), composition


def _ensure_face_model() -> str:
    """Download the YuNet face-detection model on first use (OpenCV 5.0 no longer bundles Haar cascades)."""
    model_path = config.FACE_MODEL_PATH
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(config.FACE_MODEL_URL, model_path)
    return model_path


def _build_face_detector(width: int, height: int):
    """Build a YuNet face detector sized to the frame dimensions, or None if unavailable (degrades gracefully)."""
    try:
        model_path = _ensure_face_model()
        return cv2.FaceDetectorYN_create(model_path, "", (width, height))
    except Exception:
        return None


def detect_face_box(frame) -> tuple[int, int, int, int] | None:
    """Largest face in a BGR frame as (x, y, w, h), or None. Used on generated images too."""
    h, w = frame.shape[:2]
    detector = _build_face_detector(w, h)
    if detector is None:
        return None
    try:
        _, faces = detector.detect(frame)
    except cv2.error:
        return None
    if faces is None or len(faces) == 0:
        return None
    # faces columns: x, y, w, h, then landmarks, then confidence score last
    best = max(faces, key=lambda f: f[2] * f[3])  # largest face by area
    return (int(best[0]), int(best[1]), int(best[2]), int(best[3]))


def _phash(gray) -> int:
    """64-bit perceptual hash (DCT based), used to spot near-identical frames."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
    dct = cv2.dct(small)[:8, :8]
    flat = dct.flatten()
    median = np.median(flat[1:])  # skip the DC term, it only encodes overall brightness
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def score_frames(sampled: list[tuple[str, float]]) -> list[Candidate]:
    """Score each sampled frame with cheap CV heuristics: sharpness, face, exposure, colour, composition."""
    face_detector = None
    candidates = []

    for i, (path, timestamp) in enumerate(sampled):
        frame = cv2.imread(path)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < config.MIN_LAPLACIAN_VAR:
            continue

        if face_detector is None:
            face_detector = _build_face_detector(width, height)

        face_box = None
        if face_detector is not None:
            try:
                _, faces = face_detector.detect(frame)
            except cv2.error:
                faces = None
            if faces is not None and len(faces) > 0:
                best = max(faces, key=lambda f: f[2] * f[3])
                face_box = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))

        face_fraction, face_quality, composition = _face_scores(face_box, width, height)
        sharpness_score = min(lap_var / 300.0, 1.0)
        exposure_score = _exposure_score(gray)
        colorfulness = _colorfulness(frame)

        total = (
            config.WEIGHT_SHARPNESS * sharpness_score
            + config.WEIGHT_FACE * face_quality
            + config.WEIGHT_EXPOSURE * exposure_score
            + config.WEIGHT_COLORFULNESS * colorfulness
            + config.WEIGHT_COMPOSITION * composition
        )

        candidates.append(Candidate(
            id=f"c{i:04d}",
            path=path,
            timestamp=timestamp,
            score=total,
            sharpness=sharpness_score,
            has_face=face_box is not None,
            exposure_score=exposure_score,
            colorfulness=colorfulness,
            composition=composition,
            face_fraction=face_fraction,
            face_box=face_box,
            phash=_phash(gray),
        ))

    return candidates


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def diversify(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Take the best `limit` candidates that are not near-duplicates of each other.

    Greedy: walk the frames best-first and keep one only if it looks different from
    everything kept so far *and* is far enough away in time. Without this, a talking-head
    video hands the director eight copies of the same shot and there is no real choice to
    make — the whole point of a shortlist is that the options differ.
    """
    kept: list[Candidate] = []

    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        too_similar = any(
            abs(candidate.timestamp - k.timestamp) < config.MIN_CANDIDATE_GAP_SECONDS
            or _hamming(candidate.phash, k.phash) <= config.DEDUPE_HASH_DISTANCE
            for k in kept
        )
        if not too_similar:
            kept.append(candidate)
        if len(kept) >= limit:
            break

    # A short or very static video can legitimately fail the diversity test everywhere.
    # Backfill on score rather than handing back a one-item shortlist.
    if len(kept) < limit:
        kept_ids = {c.id for c in kept}
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            if candidate.id not in kept_ids:
                kept.append(candidate)
            if len(kept) >= limit:
                break

    return kept


def get_shortlist(video_path: str, duration: float, frames_dir: str) -> list[Candidate]:
    """Full layer-1 pipeline: sample -> score -> de-duplicate -> top SHORTLIST_SIZE candidates."""
    sampled = sample_frames(video_path, duration, frames_dir)
    scored = score_frames(sampled)
    return diversify(scored, config.SHORTLIST_SIZE)
