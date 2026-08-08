"""Stage 2b — render the chosen frame into a finished thumbnail with GPT Image 2 on kie.ai.

This is image-to-image, not text-to-image, on purpose: the point of a thumbnail is that it
shows *this* video. Generating a fresh picture from a prompt would invent a stranger and
a scene that never happens, which is both a lie and, in practice, a worse click — viewers
bounce when the thumbnail does not match the first ten seconds.

The model gets the real frame plus the art direction, and is asked to grade, relight, and
declutter it while keeping the subject recognisably the same person.
"""

import os

from . import config, kie
from .director import ThumbnailPlan

NEGATIVE_SPACE = {
    "left": "Keep the subject filling the LEFT half. The RIGHT half must stay visually calm and "
            "uncluttered — a clean, darker, out-of-focus background with no competing detail.",
    "right": "Keep the subject filling the RIGHT half. The LEFT half must stay visually calm and "
             "uncluttered — a clean, darker, out-of-focus background with no competing detail.",
    "center": "Keep the subject centred in the upper two thirds. The BOTTOM third must stay "
              "visually calm and uncluttered, with no competing detail.",
}

BASE_DIRECTION = (
    "Rebuild this video still as a thumbnail a professional YouTube designer would deliver, "
    "in 16:9. The source is a raw frame from a camera; the output must look like a composed, "
    "art-directed graphic, not a screenshot with a filter over it.\n\n"

    "Keep the same person: same face, same bone structure, same skin tone, same hair, same "
    "clothing, same identity. Do not swap them for someone else, beautify them into a "
    "different face, or turn this into an illustration, 3D render or cartoon. It must still "
    "obviously be this exact person.\n\n"

    "Treat the subject as a cut-out placed onto a designed background:\n"
    "- Separate them hard from the background, the way a studio portrait does. Crisp, clean "
    "edges around hair and shoulders.\n"
    "- Light them properly: a strong key light shaping the face, a rim light along one edge "
    "to lift them off the background, and catchlights in the eyes.\n"
    "- Replace the ordinary room behind them with a clean, deliberate backdrop: a rich "
    "graduated colour, soft studio falloff, or a heavily blurred and darkened version of the "
    "original setting. It must never compete with the face.\n"
    "- Delete every trace of the original clutter: burnt-in subtitles and captions, "
    "timestamps, watermarks, channel logos, furniture edges, cables, curtain folds, doorframes "
    "and background people. Any leftover on-screen text is a failure.\n\n"

    "Finish it like a thumbnail, not like a photo:\n"
    "- Deep contrast and rich, saturated colour. Punchy, never muddy, never washed out.\n"
    "- Crisp micro-detail on the eyes, mouth and hair.\n"
    "- The face large and unmistakable at 210x118 pixels.\n"
    "- Natural skin texture. No plastic smoothing, no waxy blur, no uncanny eyes, no warped "
    "hands, no extra fingers or limbs.\n"
    "- If the mouth is caught awkwardly between syllables, settle it into a natural, "
    "deliberate expression that suits the face. It should look held for the camera, not "
    "caught by accident.\n"
)

NO_TEXT_RULE = (
    "\nCritical: render NO text, NO letters, NO numbers, NO logos, NO watermarks, and NO "
    "graphic badges anywhere in the image. The headline is composited separately, so any "
    "text you draw would collide with it."
)

TEXT_RULE = (
    '\nRender the headline "{headline}" in a heavy, bold, all-caps sans-serif with a thick '
    "outline, placed in the clear area away from the subject's face. Spell it exactly as given."
)


def build_prompt(plan: ThumbnailPlan) -> str:
    parts = [BASE_DIRECTION, NEGATIVE_SPACE[plan.subject_side]]
    if plan.scene_direction:
        parts.append(f"Art direction for this specific shot: {plan.scene_direction}")
    parts.append(TEXT_RULE.format(headline=plan.headline) if config.LET_MODEL_RENDER_TEXT else NO_TEXT_RULE)
    return "\n\n".join(part.strip() for part in parts if part.strip())


def render(frame_path: str, plan: ThumbnailPlan, dest_path: str, on_progress=None) -> str:
    """Render the chosen frame through GPT Image 2 and save the result to dest_path.

    Returns the path actually written. Raises KieError if the render fails and the caller
    has not opted into falling back to the raw frame.
    """
    frame_url = kie.upload_image(frame_path, config.RENDER_IMAGE_MAX_WIDTH)

    urls = kie.run_image_task(
        config.KIE_IMAGE_MODEL,
        {
            "prompt": build_prompt(plan),
            "input_urls": [frame_url],
            "aspect_ratio": config.KIE_IMAGE_ASPECT_RATIO,
            "resolution": config.KIE_IMAGE_RESOLUTION,
        },
        on_progress=on_progress,
    )

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    return kie.download(urls[0], dest_path)


def render_or_fallback(frame_path: str, plan: ThumbnailPlan, dest_path: str,
                       on_progress=None) -> tuple[str, str | None]:
    """render(), but degrade to the original frame instead of failing the whole run.

    Returns (path_to_use, warning). warning is None on a clean render, otherwise a
    human-readable note explaining that the output is the ungraded original frame.
    """
    try:
        return render(frame_path, plan, dest_path, on_progress=on_progress), None
    except kie.KieError as e:
        if not config.FALLBACK_TO_RAW_FRAME:
            raise
        return frame_path, f"Couldn't redraw this with GPT Image 2, so it's the original frame. {e}"
