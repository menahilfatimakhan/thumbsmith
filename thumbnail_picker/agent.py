import io
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from . import config, transcribe
from .extract import Candidate

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3

PROMPT = (
    "Here are {n} candidate frames sampled from a YouTube video, each labeled with a "
    "candidate_id and timestamp. You are designing a clickable YouTube thumbnail.\n\n"
    "1. Pick the single best frame. Prioritize a clear, expressive face or an "
    "interesting action moment over a static or dull frame. Avoid anything blurry, "
    "oddly cropped, or caught mid-motion/mid-blink. If any on-screen text/graphics "
    "in a frame are especially compelling (numbers, results, a strong claim), that's "
    "a plus.\n"
    "2. Write a short, punchy caption for the thumbnail (2-5 words, ALL CAPS, no "
    "trailing punctuation) that would make someone want to click, based on what's "
    "visible or being discussed in the frames. Do not just describe the image — "
    "write a hook, like real YouTube thumbnails use. Some candidates include a "
    "transcript excerpt of what's actually being said at that moment — when present, "
    "prefer a caption that echoes a concrete claim from the speech over a purely "
    "visual guess.\n\n"
    "Call pick_thumbnail with your choice. (Text placement is handled separately — "
    "don't worry about where the caption will go.)"
)


@dataclass
class Selection:
    candidate_id: str
    reason: str
    caption: str


def _friendly_quota_error(e: "genai_errors.ClientError") -> RuntimeError:
    """Turn Google's raw 429 error dict into a message that says what's actually wrong and what to do."""
    detail_str = str(e.details)
    if "PerDay" in detail_str:
        return RuntimeError(
            f"Gemini's free-tier daily quota for '{config.GEMINI_MODEL}' is used up for today "
            "(the free tier caps requests per day, per model). It resets roughly every 24 hours. "
            "Options: wait for the reset, switch GEMINI_MODEL in config.py to a different model "
            "(each model has its own separate quota), or enable billing on your Google AI Studio "
            "project for a much higher limit."
        )
    return RuntimeError(
        "Gemini's free-tier rate limit was hit (too many requests too quickly). Wait a short "
        "while and try again."
    )


def _resize_for_upload(path: str) -> bytes:
    img = Image.open(path).convert("RGB")
    if img.width > config.GEMINI_IMAGE_MAX_WIDTH:
        ratio = config.GEMINI_IMAGE_MAX_WIDTH / img.width
        img = img.resize((config.GEMINI_IMAGE_MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def pick_best_frame(shortlist: list[Candidate], transcript_segments: list[transcribe.Segment] | None = None) -> Selection:
    """Send the shortlisted frames to Gemini and let it pick the best frame + draft a caption.

    transcript_segments is optional (only present if OPENAI_API_KEY was set) — when given, each
    candidate is labeled with the speech happening near its timestamp, so Gemini can write a
    caption grounded in what's actually said, not just what's visible.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    result_holder: dict = {}

    def pick_thumbnail(candidate_id: str, reason: str, caption: str) -> dict:
        """Record the chosen frame and its caption.

        Args:
            candidate_id: The candidate_id of the chosen frame (e.g. "c0003").
            reason: One sentence explaining why this frame makes the best thumbnail.
            caption: A short, punchy thumbnail caption (2-5 words, ALL CAPS, no
                trailing punctuation) written as a hook, not a description.
        """
        result_holder["candidate_id"] = candidate_id
        result_holder["reason"] = reason
        result_holder["caption"] = caption
        return {"status": "recorded", "candidate_id": candidate_id}

    parts = [PROMPT.format(n=len(shortlist))]
    for c in shortlist:
        parts.append(types.Part.from_bytes(data=_resize_for_upload(c.path), mime_type="image/jpeg"))
        label = f"^ candidate_id={c.id}, timestamp={c.timestamp:.1f}s"
        near_speech = transcribe.text_near(transcript_segments or [], c.timestamp)
        if near_speech:
            label += f'\n  speech around this moment: "{near_speech}"'
        parts.append(label)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(tools=[pick_thumbnail]),
            )
            last_error = None
            break
        except genai_errors.ServerError as e:
            # 503 "model overloaded" etc. — transient, worth retrying with backoff
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except genai_errors.ClientError as e:
            if e.code == 429:
                raise _friendly_quota_error(e) from e
            raise

    if last_error is not None:
        raise RuntimeError(
            f"Gemini is temporarily overloaded after {MAX_RETRIES} attempts — try again in a bit. ({last_error})"
        )

    if "candidate_id" not in result_holder:
        raise RuntimeError("Gemini did not call pick_thumbnail — no selection was made")

    return Selection(
        candidate_id=result_holder["candidate_id"],
        reason=result_holder["reason"],
        caption=result_holder["caption"],
    )
