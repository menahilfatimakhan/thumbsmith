import os
import subprocess
from dataclasses import dataclass

from . import config


@dataclass
class Segment:
    start: float
    end: float
    text: str


def is_available() -> bool:
    """Transcription is entirely optional — only runs if the user has set an OpenAI key."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def _extract_audio(video_path: str, audio_path: str) -> None:
    """Pull a small mono audio-only file out of the video, to stay well under Whisper's 25MB limit."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "64k",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe(video_path: str) -> list[Segment]:
    """Transcribe the video's speech with OpenAI Whisper. Returns [] if unavailable or on any failure —
    this is a nice-to-have context enrichment, never something that should break the pipeline."""
    if not is_available():
        return []

    try:
        from openai import OpenAI
    except ImportError:
        return []

    audio_path = os.path.join(os.path.dirname(video_path), "audio.mp3")

    try:
        _extract_audio(video_path, audio_path)

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=config.TRANSCRIBE_MODEL,
                file=f,
                response_format="verbose_json",
            )

        segments = []
        for seg in getattr(response, "segments", None) or []:
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            text = getattr(seg, "text", None)
            if start is None or end is None or not text:
                continue
            segments.append(Segment(start=float(start), end=float(end), text=text.strip()))
        return segments

    except Exception:
        # Transcription is optional context, not a required stage — never let it take
        # down a run that would otherwise succeed on visuals alone.
        return []


def text_near(segments: list[Segment], timestamp: float, window: float = None) -> str:
    """Concatenate transcript text spoken within `window` seconds of `timestamp`."""
    if not segments:
        return ""
    window = config.TRANSCRIBE_WINDOW_SECONDS if window is None else window
    nearby = [s.text for s in segments if s.end >= timestamp - window and s.start <= timestamp + window]
    return " ".join(nearby).strip()
