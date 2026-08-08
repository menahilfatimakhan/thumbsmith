import os
import time
import traceback
import uuid

from flask import Flask, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from thumbnail_picker import config, download, kie, pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

OUTPUT_DIR = os.path.join(app.static_folder, "outputs")
INCOMING_DIR = os.path.join(config.WORK_DIR, "_incoming")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _readable_extensions() -> str:
    """"MP4, MOV, MKV, WEBM, AVI or M4V" — the raw set reads like machine output on screen."""
    common = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"]
    ordered = [e for e in common if e in config.ALLOWED_UPLOAD_EXTENSIONS]
    ordered += sorted(config.ALLOWED_UPLOAD_EXTENSIONS - set(ordered))
    names = [e.lstrip(".").upper() for e in ordered]
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} or {names[-1]}"


def _page(**kwargs):
    blocked = download.youtube_is_blocked()
    # Once YouTube has bot-blocked this host, every later link fails identically. Send
    # people to the tab that works rather than letting them retry a dead path.
    if blocked and "active_tab" not in kwargs:
        kwargs["active_tab"] = "upload"
    return render_template(
        "index.html",
        max_upload_mb=int(config.MAX_UPLOAD_BYTES / 1024 / 1024),
        allowed_extensions=_readable_extensions(),
        youtube_blocked=blocked,
        **kwargs,
    )


def _save_upload(file_storage) -> str:
    """Stream the uploaded video to a temp path in the work dir and return it."""
    os.makedirs(INCOMING_DIR, exist_ok=True)
    extension = os.path.splitext(secure_filename(file_storage.filename))[1].lower()
    temp_path = os.path.join(INCOMING_DIR, f"{uuid.uuid4().hex}{extension}")
    file_storage.save(temp_path)
    return temp_path


def _resolve_source(youtube_url: str, uploaded):
    """Turn whichever input the user supplied into a VideoSource."""
    if uploaded is not None and uploaded.filename:
        temp_path = _save_upload(uploaded)
        try:
            return download.ingest_local_video(temp_path, uploaded.filename, move=True)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return download.download_video(youtube_url)


@app.route("/", methods=["GET"])
def index():
    return _page()


@app.route("/generate", methods=["POST"])
def generate():
    youtube_url = request.form.get("youtube_url", "").strip()
    uploaded = request.files.get("video_file")
    has_upload = uploaded is not None and bool(uploaded.filename)
    active_tab = "upload" if has_upload else "link"

    if not youtube_url and not has_upload:
        return _page(error="Paste a YouTube link or choose a video file to upload.")

    try:
        source = _resolve_source(youtube_url, uploaded)

        output_filename = f"{source.video_id}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        outcome = pipeline.generate_thumbnail(source, output_path)

        result = {
            "image_url": url_for("static", filename=f"outputs/{output_filename}") + f"?v={int(time.time())}",
            "title": source.title,
            "origin": source.origin,
            "headline": outcome.plan.headline,
            "timestamp": round(outcome.chosen.timestamp, 1),
            "reason": outcome.plan.reason,
            "text_side": outcome.text_side,
            "rendered": outcome.rendered,
            "transcribed": outcome.transcribed,
            "warnings": outcome.warnings,
            "image_model": config.KIE_IMAGE_MODEL,
            "chat_model": config.KIE_CHAT_MODEL,
        }
        return _page(youtube_url=youtube_url, result=result, active_tab=active_tab)

    except (kie.KieError, RuntimeError) as e:
        # Missing/invalid API key, out of credits, unusable video, model refused to answer.
        return _page(youtube_url=youtube_url, error=str(e), active_tab=active_tab)
    except Exception as e:
        traceback.print_exc()
        return _page(youtube_url=youtube_url, error=f"{type(e).__name__}: {e}", active_tab=active_tab)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    return _page(
        error=f"That video is over the {int(config.MAX_UPLOAD_BYTES / 1024 / 1024)} MB upload limit. "
              "Export it smaller, trim it, or paste a YouTube link instead.",
        active_tab="upload",
    ), 413


if __name__ == "__main__":
    if not os.environ.get(config.KIE_API_KEY_ENV):
        print(f"WARNING: {config.KIE_API_KEY_ENV} is not set. Get a key at https://kie.ai/api-key, then:")
        print(f'  $env:{config.KIE_API_KEY_ENV} = "your-key"')
    # threaded=True so one long-running /generate request doesn't freeze the whole server —
    # without it, the dev server is single-threaded and even the homepage becomes unreachable
    # while a thumbnail is being generated.
    app.run(debug=True, port=5000, threaded=True)
