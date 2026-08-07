import glob
import os
import subprocess
import urllib.request
from dataclasses import dataclass

import cv2

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
    face_box: tuple[int, int, int, int] | None  # (x, y, w, h) in the original frame's pixel space


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


def _exposure_score(gray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_px = gray.size
    dark_frac = hist[:16].sum() / total_px
    bright_frac = hist[240:].sum() / total_px
    mean_brightness = gray.mean()
    brightness_closeness = 1.0 - abs(mean_brightness - 125) / 125
    return max(0.0, min(brightness_closeness - (dark_frac + bright_frac), 1.0))


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


def score_frames(sampled: list[tuple[str, float]]) -> list[Candidate]:
    """Score each sampled frame with cheap CV heuristics: sharpness, face presence, exposure."""
    face_detector = None

    candidates = []
    for i, (path, timestamp) in enumerate(sampled):
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < config.MIN_LAPLACIAN_VAR:
            continue

        if face_detector is None:
            h, w = frame.shape[:2]
            face_detector = _build_face_detector(w, h)

        face_box = None
        if face_detector is not None:
            _, faces = face_detector.detect(frame)
            if faces is not None and len(faces) > 0:
                # faces columns: x, y, w, h, then landmarks, then confidence score last
                best = max(faces, key=lambda f: f[2] * f[3])  # largest face by area
                face_box = (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
        has_face = face_box is not None

        sharpness_score = min(lap_var / 300.0, 1.0)
        exposure_score = _exposure_score(gray)

        total = (
            config.WEIGHT_SHARPNESS * sharpness_score
            + config.WEIGHT_FACE * (1.0 if has_face else 0.0)
            + config.WEIGHT_EXPOSURE * exposure_score
        )

        candidates.append(Candidate(
            id=f"c{i:04d}",
            path=path,
            timestamp=timestamp,
            score=total,
            sharpness=sharpness_score,
            has_face=has_face,
            exposure_score=exposure_score,
            face_box=face_box,
        ))

    return candidates


def get_shortlist(video_path: str, duration: float, frames_dir: str) -> list[Candidate]:
    """Full layer-1 pipeline: sample -> score -> return the top SHORTLIST_SIZE candidates."""
    sampled = sample_frames(video_path, duration, frames_dir)
    scored = score_frames(sampled)
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[: config.SHORTLIST_SIZE]
