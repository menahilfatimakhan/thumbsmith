"""Stage 2a — the art director.

A vision model on kie.ai looks at every shortlisted frame and returns a *plan* rather
than just a pick: which frame, the headline to put on it, which side the subject should
sit on, the accent colour, and the art direction to hand the image model.

Everything downstream (the render, the typography) reads from that one plan, which is
what keeps the caption, the negative space, and the text placement agreeing with each
other instead of each stage guessing independently.
"""

import json
import re
from dataclasses import dataclass

from . import config, kie, transcribe
from .extract import Candidate

VALID_SIDES = ("left", "right", "center")
MAX_HEADLINE_CHARS = 28

SYSTEM_PROMPT = (
    "You are a senior YouTube thumbnail art director. You have designed thumbnails for "
    "channels with tens of millions of views. You know that a thumbnail is judged at "
    "210x118 pixels on a phone, in competition with eleven others, in under a second.\n\n"
    "You reply with a single JSON object and nothing else. No prose, no markdown fences."
)

BRIEF = """\
Below are {n} candidate frames sampled from a video{title_clause}. Each frame is labelled \
with a candidate_id, its timestamp, and sometimes the speech happening at that moment.

Design the thumbnail.

FRAME CHOICE — pick the one frame that will earn the click:
- A large, clearly readable face with a strong, specific emotion beats everything else. \
Shock, delight, disbelief, strain, focus. A neutral or mid-blink face is worthless.
- If no face works, pick the most legible single moment of action, contrast, or result. \
Reject anything blurry, dim, cluttered, or mid-motion.
- Prefer a frame with room around the subject. A frame where the subject is dead centre \
and edge-to-edge leaves nowhere for the headline.
- On-screen numbers, results, or a visible before/after are a strong plus.

HEADLINE — 2 to 4 words, {max_chars} characters maximum, ALL CAPS, no trailing punctuation:
- Open a curiosity gap or make a concrete claim. "IT ALL BROKE", "$0 TO $40K", "NOBODY \
EXPECTED THIS".
- Never describe the picture. "MAN AT DESK" is a caption; you are writing a hook.
- If speech context is provided, anchor the headline in something actually said or shown. \
Specific beats vague. Real numbers beat adjectives.
- It must stay readable at thumbnail size, so short words win.

LAYOUT — subject_side is where the human or main subject should sit in the final image. \
The headline goes on the opposite side, so never say "center" unless the subject genuinely \
cannot be moved off the middle.

Return exactly this JSON object:

{{
  "candidate_id": "the id of the frame you chose",
  "reason": "one sentence on why this frame wins",
  "headline": "THE HEADLINE",
  "accent_word": "the single word or number from the headline to colour for emphasis, or \\"\\" for none",
  "subject_side": "left" | "right" | "center",
  "accent_color": "#RRGGBB, high-contrast against the frame",
  "scene_direction": "one or two sentences of art direction for an image model that will \
re-render this exact frame as a polished thumbnail: what to keep, what clutter to remove, \
how to light the subject, what the background should become. Never mention text or words."
}}"""


@dataclass
class ThumbnailPlan:
    candidate_id: str
    reason: str
    headline: str
    accent_word: str
    subject_side: str
    accent_color: str
    scene_direction: str

    @property
    def text_side(self) -> str:
        """Where the headline goes: the side the subject is not on."""
        return {"left": "right", "right": "left", "center": "bottom"}[self.subject_side]


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a model reply that may be fenced or padded with prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except ValueError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"no JSON object found in model reply: {text[:200]}")


def _clean_headline(raw: str) -> str:
    headline = re.sub(r"\s+", " ", str(raw or "")).strip().strip('"\'').rstrip(".!,;:")
    headline = headline.upper()
    if len(headline) > MAX_HEADLINE_CHARS:
        # Trim on a word boundary rather than mid-word — a clipped word looks like a bug.
        words = headline.split()
        headline = ""
        for word in words:
            if len(f"{headline} {word}".strip()) > MAX_HEADLINE_CHARS:
                break
            headline = f"{headline} {word}".strip()
    return headline


def _clean_color(raw: str) -> str:
    value = str(raw or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value.upper()
    if re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return f"#{value.upper()}"
    return config.DEFAULT_ACCENT_COLOR


def _to_plan(data: dict, valid_ids: set[str]) -> ThumbnailPlan:
    candidate_id = str(data.get("candidate_id", "")).strip()
    if candidate_id not in valid_ids:
        raise ValueError(f"model chose candidate_id={candidate_id!r}, which is not in the shortlist")

    headline = _clean_headline(data.get("headline"))
    if not headline:
        raise ValueError("model returned an empty headline")

    # Only keep the accent word if it really is one of the headline's words — otherwise
    # the compositor would highlight nothing and silently drop the emphasis.
    accent_word = _clean_headline(data.get("accent_word"))
    if accent_word not in headline.split():
        accent_word = ""

    subject_side = str(data.get("subject_side", "")).strip().lower()
    if subject_side not in VALID_SIDES:
        subject_side = "center"

    return ThumbnailPlan(
        candidate_id=candidate_id,
        reason=str(data.get("reason", "")).strip() or "Selected as the strongest frame.",
        headline=headline,
        accent_word=accent_word,
        subject_side=subject_side,
        accent_color=_clean_color(data.get("accent_color")),
        scene_direction=str(data.get("scene_direction", "")).strip(),
    )


def _build_messages(shortlist: list[Candidate], frame_urls: dict[str, str],
                    transcript_segments: list[transcribe.Segment], title: str) -> list[dict]:
    title_clause = f' titled "{title}"' if title else ""
    content: list[dict] = [{
        "type": "text",
        "text": BRIEF.format(n=len(shortlist), title_clause=title_clause, max_chars=MAX_HEADLINE_CHARS),
    }]

    for candidate in shortlist:
        content.append({"type": "image_url", "image_url": {"url": frame_urls[candidate.id]}})
        label = f"^ candidate_id={candidate.id}, timestamp={candidate.timestamp:.1f}s"
        speech = transcribe.text_near(transcript_segments or [], candidate.timestamp)
        if speech:
            label += f'\n  speech around this moment: "{speech}"'
        content.append({"type": "text", "text": label})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def direct(shortlist: list[Candidate], frame_urls: dict[str, str],
           transcript_segments: list[transcribe.Segment] | None = None,
           title: str = "") -> ThumbnailPlan:
    """Ask the kie.ai vision model to turn the shortlist into a full thumbnail plan.

    frame_urls maps candidate_id -> the public URL that frame was uploaded to.
    transcript_segments is optional (only present if OPENAI_API_KEY was set) — when given,
    each candidate is labelled with the speech near its timestamp, so the headline can be
    grounded in what is actually said rather than guessed from pixels.
    """
    messages = _build_messages(shortlist, frame_urls, transcript_segments or [], title)
    valid_ids = {c.id for c in shortlist}
    last_error: Exception | None = None

    for attempt in range(2):
        reply = kie.chat(messages)
        try:
            return _to_plan(_extract_json(reply), valid_ids)
        except ValueError as e:
            last_error = e
            # Show the model its own bad output and ask again. One retry only — if it
            # cannot produce valid JSON twice, retrying further just burns credits.
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    f"That was not usable: {e}. Reply again with ONLY the JSON object, "
                    f"using a candidate_id from this list: {sorted(valid_ids)}."
                )},
            ]

    raise kie.KieError(
        f"The model couldn't settle on a usable plan after two tries. ({last_error})"
    )
