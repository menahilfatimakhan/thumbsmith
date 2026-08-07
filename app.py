import os
import time
import traceback

from flask import Flask, render_template, request, url_for

from thumbnail_picker import agent, compose, config, download, extract, transcribe

app = Flask(__name__)

OUTPUT_DIR = os.path.join(app.static_folder, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    youtube_url = request.form.get("youtube_url", "").strip()

    if not youtube_url:
        return render_template("index.html", error="Please paste a YouTube link.")

    try:
        dl = download.download_video(youtube_url)

        frames_dir = os.path.join(os.path.dirname(dl.video_path), "frames")
        shortlist = extract.get_shortlist(dl.video_path, dl.duration, frames_dir)
        if not shortlist:
            return render_template(
                "index.html",
                youtube_url=youtube_url,
                error="No usable frames passed the sharpness/exposure filter for this video. Try a different one.",
            )

        transcript_segments = transcribe.transcribe(dl.video_path) if transcribe.is_available() else []

        selection = agent.pick_best_frame(shortlist, transcript_segments)
        chosen = next(c for c in shortlist if c.id == selection.candidate_id)

        output_filename = f"{dl.video_id}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        compose.make_thumbnail(chosen.path, output_path, selection.caption, chosen.face_box)

        image_url = url_for("static", filename=f"outputs/{output_filename}") + f"?v={int(time.time())}"

        result = {
            "image_url": image_url,
            "title": dl.title,
            "caption": selection.caption,
            "timestamp": round(chosen.timestamp, 1),
            "reason": selection.reason,
            "transcribed": bool(transcript_segments),
        }
        return render_template("index.html", youtube_url=youtube_url, result=result)

    except RuntimeError as e:
        # e.g. missing GEMINI_API_KEY, or Gemini not calling the tool
        return render_template("index.html", youtube_url=youtube_url, error=str(e))
    except Exception as e:
        traceback.print_exc()
        return render_template(
            "index.html",
            youtube_url=youtube_url,
            error=f"{type(e).__name__}: {e}",
        )


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY is not set. Set it before generating a thumbnail:")
        print('  $env:GEMINI_API_KEY = "your-key"')
    # threaded=True so one long-running /generate request doesn't freeze the whole server —
    # without it, the dev server is single-threaded and even the homepage becomes unreachable
    # while a thumbnail is being generated.
    app.run(debug=True, port=5000, threaded=True)
