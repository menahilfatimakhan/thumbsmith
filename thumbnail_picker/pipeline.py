"""The whole run, start to finish, in one place.

``app.py`` and ``main.py`` are both thin wrappers around ``generate_thumbnail`` — the web
form and the CLI differ only in how they get a video and how they report progress, never
in what the pipeline does.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import compose, config, director, extract, generate, kie, transcribe
from .download import VideoSource


@dataclass
class ThumbnailResult:
    output_path: str
    source: VideoSource
    plan: director.ThumbnailPlan
    chosen: extract.Candidate
    shortlist: list[extract.Candidate] = field(default_factory=list)
    transcribed: bool = False
    rendered: bool = False       # True if GPT Image 2 produced the picture
    text_side: str = "bottom"
    warnings: list[str] = field(default_factory=list)


def _upload_shortlist(shortlist: list[extract.Candidate]) -> dict[str, str]:
    """Upload every shortlisted frame and return {candidate_id: public_url}.

    In parallel because these are eight independent round trips and doing them one after
    another is the single slowest non-model step in the run.
    """
    with ThreadPoolExecutor(max_workers=min(8, len(shortlist))) as pool:
        urls = pool.map(
            lambda c: kie.upload_image(c.path, config.DIRECTOR_IMAGE_MAX_WIDTH),
            shortlist,
        )
        return {c.id: url for c, url in zip(shortlist, urls)}


def generate_thumbnail(source: VideoSource, output_path: str, on_step=None) -> ThumbnailResult:
    """Turn a downloaded or uploaded video into a finished thumbnail at output_path."""
    def step(message: str) -> None:
        if on_step:
            on_step(message)

    warnings: list[str] = []

    frames_dir = os.path.join(os.path.dirname(source.video_path), "frames")
    step("Sampling and scoring frames...")
    shortlist = extract.get_shortlist(source.video_path, source.duration, frames_dir)
    if not shortlist:
        raise RuntimeError(
            "No usable frames passed the sharpness and exposure filter for this video. "
            "It may be very dark, very blurry, or almost entirely static."
        )

    transcript_segments = []
    if transcribe.is_available():
        step("Transcribing audio for headline context...")
        transcript_segments = transcribe.transcribe(source.video_path)
        if not transcript_segments:
            warnings.append("Transcription returned nothing — the headline is based on visuals alone.")

    step(f"Uploading {len(shortlist)} candidate frames to kie.ai...")
    frame_urls = _upload_shortlist(shortlist)

    step(f"Asking {config.KIE_CHAT_MODEL} to art-direct the thumbnail...")
    plan = director.direct(shortlist, frame_urls, transcript_segments, source.title)
    chosen = next(c for c in shortlist if c.id == plan.candidate_id)

    step(f"Rendering with {config.KIE_IMAGE_MODEL}...")
    render_dir = os.path.join(os.path.dirname(source.video_path), "render")
    render_path = os.path.join(render_dir, f"{plan.candidate_id}.png")
    image_path, render_warning = generate.render_or_fallback(
        chosen.path, plan, render_path,
        on_progress=lambda state, pct: step(f"  render {state}{f' {pct}%' if pct else ''}"),
    )
    rendered = render_warning is None
    if render_warning:
        warnings.append(render_warning)

    step("Composing the headline...")
    layout = compose.make_thumbnail(
        image_path, output_path,
        headline=plan.headline,
        accent_word=plan.accent_word,
        accent_color=plan.accent_color,
        preferred_side=plan.text_side,
        # GPT Image 2 already delivers a graded picture; grading it again crushes it.
        grade_strength=0.25 if rendered else 1.0,
    )
    if not layout["face_detected"]:
        warnings.append("No face found in the final image — the headline was placed on composition alone.")

    return ThumbnailResult(
        output_path=output_path,
        source=source,
        plan=plan,
        chosen=chosen,
        shortlist=shortlist,
        transcribed=bool(transcript_segments),
        rendered=rendered,
        text_side=layout["text_side"],
        warnings=warnings,
    )
