# How It Works

This document explains, in depth, how this project turns a YouTube link into a designed thumbnail — what each file does, why it's built the way it is, and the real bugs we hit and fixed along the way.

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Why Two Layers? (The Core Design Decision)](#2-why-two-layers-the-core-design-decision)
3. [Project File Map](#3-project-file-map)
4. [Stage-by-Stage Walkthrough](#4-stage-by-stage-walkthrough)
   - [4.1 Download (`download.py`)](#41-download-downloadpy)
   - [4.2 Frame Sampling (`extract.py` — part 1)](#42-frame-sampling-extractpy--part-1)
   - [4.3 Heuristic Scoring (`extract.py` — part 2)](#43-heuristic-scoring-extractpy--part-2)
   - [4.3.5 (Optional) Speech Transcription (`transcribe.py`)](#435-optional-speech-transcription-transcribepy)
   - [4.4 The Gemini Agent (`agent.py`)](#44-the-gemini-agent-agentpy)
   - [4.5 Composing the Final Thumbnail (`compose.py`)](#45-composing-the-final-thumbnail-composepy)
5. [Two Ways to Run It: CLI vs. Web App](#5-two-ways-to-run-it-cli-vs-web-app)
   - [5.1 CLI (`main.py`)](#51-cli-mainpy)
   - [5.2 Web App (`app.py` + `templates/index.html`)](#52-web-app-apppy--templatesindexhtml)
6. [Configuration Reference (`config.py`)](#6-configuration-reference-configpy)
7. [Real Bugs We Hit and How We Fixed Them](#7-real-bugs-we-hit-and-how-we-fixed-them)
8. [What's Deliberately Not Here (Stretch Goals)](#8-whats-deliberately-not-here-stretch-goals)

---

## 1. The Big Picture

**Input:** a YouTube URL.
**Output:** a designed 1280×720 thumbnail JPEG — a good frame, color-punched, with a bold caption overlaid.

The pipeline has six steps, run in a straight line:

```
YouTube URL
    │
    ▼
① Download video (yt-dlp)              — thumbnail_picker/download.py
    │
    ▼
② Sample ~100-150 frames (ffmpeg)       — thumbnail_picker/extract.py
    │
    ▼
③ Score every frame with cheap CV       — thumbnail_picker/extract.py
   (sharpness, face presence, exposure)
    │
    ▼  (top 8 candidates only)
③.5 (optional) Transcribe the audio     — thumbnail_picker/transcribe.py
    with OpenAI Whisper, if OPENAI_API_KEY is set
    │
    ▼
④ Gemini looks at the 8 (+ nearby        — thumbnail_picker/agent.py
   speech, if transcribed) and picks
   the best one + writes a caption
    │
    ▼
⑤ Crop, resize, color-boost, and        — thumbnail_picker/compose.py
   overlay the caption text
    │
    ▼
thumbnail.jpg
```

Steps ①–③ and ⑤ are **plain deterministic code** — no AI, no API calls, free, fast, and 100% reproducible. Step ④ is the **one place an actual AI model is involved** (with an optional second AI call at ③.5 — see below), and it's involved because it's doing something heuristics genuinely can't: judging which frame is *compelling*, not just technically "correct."

---

## 2. Why Two Layers? (The Core Design Decision)

The very first question this project raises is: **do you need an AI agent at all, or can you just write scoring math?**

The answer we landed on: **both, but for different jobs.**

- **Classical computer vision (OpenCV) is great at objective, measurable properties** of an image: Is it blurry? Is it too dark? Is there a face in it? These are things you can compute with a formula, and a formula is faster, free, and perfectly consistent.
- **Classical computer vision is bad at *taste*.** It has no way to know that a frame with "$3,459 saved" written on screen is more clickable than a frame that's merely sharp and well-lit. That requires actually understanding what's happening in the image and reasoning about what a human would click on — which is exactly what a vision-capable LLM is good at.

So the architecture is:

| Layer | What it does | Technology | Cost |
|---|---|---|---|
| **Layer 1 — pre-filter** | Downloads the video, samples ~100-150 frames, scores each with sharpness/face/exposure math, keeps only the top 8 | Plain Python + OpenCV + ffmpeg | Free, no API calls |
| **Layer 2 — the agent** | Looks at those 8 images and picks the one a human would actually click on, then writes a caption | Google Gemini (vision + function calling) | Free tier (rate-limited) |

**Why not skip Layer 1 and just show Gemini all 150 frames?** Two reasons: (1) most of those 150 frames are near-duplicates or obviously bad — blurry, mid-blink, transition frames — and there's no reason to spend API quota/tokens on them, and (2) the free tier has real rate limits, so keeping the image count small (8, not 150) keeps every run cheap and fast.

**Why not skip Gemini and just use the heuristic score directly?** Because the heuristic score has no idea what's *on screen*. Every frame in our test video scored almost identically on sharpness/face/exposure (they're all clips of the same well-lit talking-head shot) — the heuristic literally cannot tell them apart. Gemini was the only part of the pipeline that noticed one specific frame had a compelling "$3,459 saved" graphic on it, which is the actual reason a viewer would click.

This is the general pattern worth remembering: **use cheap deterministic filtering to narrow down a large search space, then use the expensive/smart model only on the small shortlist where judgment actually matters.**

---

## 3. Project File Map

```
thumbnail_picker/
    __init__.py         (empty — makes this a Python package)
    config.py           All tunable constants live here
    download.py         YouTube URL -> local video file (yt-dlp)
    extract.py          video file -> sampled frames -> heuristic-scored shortlist
    transcribe.py       (optional) video file -> speech transcript, via OpenAI Whisper
    agent.py            shortlist (+ transcript) -> Gemini picks best frame + writes caption
    compose.py          chosen frame + caption -> final designed thumbnail JPEG
main.py                 CLI entrypoint (wires the above together)
app.py                  Flask web app (same pipeline, browser front-end)
templates/
    index.html          The web UI: URL input form + result display
static/
    outputs/            Generated thumbnails served by the web app
models/
    face_detection_yunet_2023mar.onnx   Face-detection model (auto-downloaded)
work/                   Scratch space: downloaded videos + sampled frames per video ID
requirements.txt        pip dependencies
```

Nothing here is a black box — every stage is a plain Python function you can call directly and inspect, which is why we could test each stage independently (see the debugging story in [§7](#7-real-bugs-we-hit-and-how-we-fixed-them)).

---

## 4. Stage-by-Stage Walkthrough

### 4.1 Download (`download.py`)

```python
def download_video(url: str) -> DownloadResult
```

Uses **`yt-dlp`** (a YouTube-downloader library, actively maintained fork of the old `youtube-dl`) via its Python API — not the command line — so failures raise catchable Python exceptions instead of just printing to a terminal.

Key detail: the format string caps the download at 720p —

```python
"format": "bestvideo[height<=720]+bestaudio/best[height<=720]"
```

There's no reason to pull a 4K source video when the final output is only 1280×720 — it would just waste bandwidth and disk space. Video and audio streams are downloaded separately and merged into one `.mp4` file, which is why **ffmpeg is a required external dependency** (yt-dlp shells out to it for the merge).

The result is saved to `work/<video_id>/source.mp4` and the function returns a `DownloadResult` (path, video ID, title, duration) that every later stage needs.

### 4.2 Frame Sampling (`extract.py` — part 1)

```python
def sample_frames(video_path, duration, frames_dir) -> list[(path, timestamp)]
```

This calls `ffmpeg` as a subprocess to pull frames out of the video at a fixed interval:

```
ffmpeg -ss <skip> -i source.mp4 -t <trimmed_duration> -vf "fps=1/interval" frames/frame_%04d.jpg
```

Two things worth understanding:

**1. The interval adapts to video length**, capped so no video ever produces more than `MAX_SAMPLED_FRAMES` (150) frames — see `_sample_interval()`:
```python
def _sample_interval(duration):
    if natural_count <= max_count:
        return SAMPLE_INTERVAL_SECONDS   # normally 1 frame every 2 seconds
    return duration / max_count          # widen the gap for very long videos
```
A 10-minute video samples every 2 seconds (~300 frames → capped at 150, so really every 4 seconds). A 2-hour video would sample roughly every 48 seconds. This bounds how long the scoring step (§4.3) takes regardless of video length.

**2. The first/last 3% of the video is skipped** (`SKIP_INTRO_OUTRO_FRACTION`), since intros, outros, and end-cards are usually low-value thumbnail material.

### 4.3 Heuristic Scoring (`extract.py` — part 2)

```python
def score_frames(sampled) -> list[Candidate]
def get_shortlist(video_path, duration, frames_dir) -> list[Candidate]   # sample + score + top 8
```

For every sampled frame, three independent signals are computed and combined into one score:

**Sharpness** — via the Laplacian variance of the grayscale image. A Laplacian highlights edges; a sharp, in-focus image has lots of strong edges (high variance), while a blurry or motion-smeared frame has weak, muddy edges (low variance).
```python
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
sharpness_score = min(lap_var / 300.0, 1.0)   # 300 ≈ empirically "clearly sharp"
```
Frames below `MIN_LAPLACIAN_VAR` (20) are rejected outright as unusable — a hard floor, not just a low score.

**Face presence** — a face detector runs on every frame and, if it finds one, records both `has_face = True` **and the face's exact pixel bounding box** (`face_box = (x, y, w, h)`, the largest face if there are several). That bounding box isn't just used for scoring — it's what Stage ⑤ uses later to place the caption without covering the face (see [§4.5](#45-composing-the-final-thumbnail-composepy)). (See [§7.1](#71-opencv-50-deleted-the-classic-face-detector) for why this isn't the simple `cv2.CascadeClassifier` you'd expect from most tutorials.)

**Exposure** — via the grayscale histogram: what fraction of pixels are near-black or near-white ("dead" pixels with no detail), and how close the average brightness is to a well-exposed midpoint (125 out of 255).
```python
brightness_closeness = 1.0 - abs(mean_brightness - 125) / 125
exposure_score = brightness_closeness - (dark_frac + bright_frac)   # clamped to [0, 1]
```

These three combine as a **weighted sum**:
```python
total = 0.4 * sharpness_score + 0.4 * has_face + 0.2 * exposure_score
```
Sharpness and face presence are weighted equally and heaviest, because "is this frame usable at all" and "is there a person in it" are the two dominant signals for a typical talking-head/tutorial video. Exposure is a lighter-weight tiebreaker. These weights are constants in `config.py`, meant to be tuned by eye rather than treated as gospel.

Every scored candidate is sorted by `total` descending, and only the **top 8** (`SHORTLIST_SIZE`) move on to the Gemini stage.

### 4.3.5 (Optional) Speech Transcription (`transcribe.py`)

```python
def transcribe(video_path: str) -> list[Segment]      # Segment = (start, end, text)
def text_near(segments, timestamp, window=6.0) -> str
```

This stage is **entirely optional** — it only runs if the `OPENAI_API_KEY` environment variable is set. Without it, `transcribe.is_available()` returns `False`, every caller skips straight past this stage, and the pipeline behaves exactly as if this file didn't exist. This matters because, unlike Gemini's free tier, OpenAI's transcription API has no persistent free quota — it's genuinely cheap (~$0.006/minute of audio, a few cents per video) but it is real billed usage, so it shouldn't be a silent requirement.

Why it exists: Gemini's frame-picking and captioning (§4.4) only ever *sees* the candidate frames — it has no idea what the speaker is actually saying at that moment. A caption grounded in an actual spoken claim ("I saved $3,459 doing this") is a stronger hook than one guessed purely from what's visible. Transcription closes that gap.

**How it works:**
1. **Extract audio only, not the whole video** (`_extract_audio`, via ffmpeg: `-vn -acodec libmp3lame -b:a 64k -ar 16000 -ac 1`). OpenAI's transcription endpoint caps uploads at 25MB — a mono 16kHz 64kbps MP3 keeps even a long video's audio track well under that (roughly 0.5MB per minute), where sending the full merged video file would risk blowing past the limit on anything more than a few minutes long.
2. **Transcribe with OpenAI Whisper** (`client.audio.transcriptions.create(model="whisper-1", response_format="verbose_json")`), which returns not just the full text but **timestamped segments** — exactly what's needed to answer "what was being said around second 493.2?" rather than just "what was said somewhere in this video?"
3. **`text_near(segments, timestamp, window=6.0)`** concatenates every segment whose time range overlaps a ±6 second window around a candidate frame's timestamp — a small, relevant excerpt, not the whole transcript.
4. **Every failure mode degrades to "no transcript," never a crash.** Missing key, missing `openai` package, network failure, ffmpeg failure, malformed response — all caught and turned into `[]`. Transcription is context enrichment, not a required stage; a run should never fail because this optional nice-to-have didn't work.

**How it plugs into the Gemini call (§4.4):** `main.py`/`app.py` call `transcribe.transcribe(video_path)` once (if available) and pass the resulting segments into `agent.pick_best_frame(shortlist, transcript_segments)`. Inside `agent.py`, each candidate's text label gets an extra line when relevant speech was found nearby:
```python
label = f"^ candidate_id={c.id}, timestamp={c.timestamp:.1f}s"
near_speech = transcribe.text_near(transcript_segments or [], c.timestamp)
if near_speech:
    label += f'\n  speech around this moment: "{near_speech}"'
```
The prompt tells Gemini explicitly to prefer a caption that echoes a concrete spoken claim over a purely visual guess when that context is present.

### 4.4 The Gemini Agent (`agent.py`)

```python
def pick_best_frame(shortlist: list[Candidate], transcript_segments: list[Segment] | None = None) -> Selection
```

This is the one *required* AI-model call in the whole pipeline (transcription in §4.3.5 is an optional second one). Here's exactly what happens:

**1. Images are resized down before upload** (`_resize_for_upload`, capped at 800px wide). Vision token cost scales with image resolution, so there's no reason to send full-resolution frames just to have Gemini glance at them.

**2. All 8 images + a text prompt go into one request**, as a `contents` list mixing plain strings and `types.Part.from_bytes(...)` image blocks — this is Gemini's standard way of taking multiple images in a single call:
```python
parts = [PROMPT.format(n=8)]
for c in shortlist:
    parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    parts.append(f"^ candidate_id={c.id}, timestamp={c.timestamp:.1f}s")
```
Each image is immediately followed by a text label identifying its `candidate_id`, so Gemini can refer back to a specific frame unambiguously in its answer.

**3. One tool does two jobs at once.** Instead of asking Gemini to write free-form text and then trying to parse an answer out of it (fragile), we give it exactly one function to call, and its parameters *are* the structured output we want:

```python
def pick_thumbnail(candidate_id: str, reason: str, caption: str) -> dict:
    """Record the chosen frame and its caption."""
```

The **prompt** tells Gemini to do both things — pick the best frame, and write a punchy 2–5 word ALL-CAPS caption (a hook, not a description) — and then call `pick_thumbnail` with both values.

Earlier versions of this pipeline also asked Gemini to guess whether the caption should go at the `"top"` or `"bottom"` of the frame. **We removed that** — see [§4.5](#45-composing-the-final-thumbnail-composepy) for why letting Gemini guess placement from a downsized image was the wrong tool for that job, and what replaced it.

This uses the `google-genai` SDK's **automatic function calling**: you pass a plain Python function (its docstring becomes the tool description Gemini sees, its type hints become the parameter schema), and the SDK executes it automatically the moment Gemini decides to call it. There's no manual "parse the tool call, run it myself, send the result back" loop to write — which is genuinely simpler than the equivalent pattern on some other model providers.

**4. Getting the result back out.** Since `pick_thumbnail`'s actual *return value* isn't what we care about (we care about its *arguments*), the function captures them into a closure variable (`result_holder`) at the moment it's called:
```python
def pick_thumbnail(candidate_id, reason, caption):
    result_holder["candidate_id"] = candidate_id
    ...
```
After `generate_content()` returns, we read the selection out of `result_holder` rather than trying to parse Gemini's final text response.

**5. Retries with backoff.** Gemini's free tier occasionally returns a transient `503 UNAVAILABLE` ("model experiencing high demand"). We hit this for real during testing — twice in a row — so the call is wrapped in up to 3 attempts with increasing backoff (3s, 6s) before giving up and surfacing a clear error message.

### 4.5 Composing the Final Thumbnail (`compose.py`)

```python
def make_thumbnail(frame_path, output_path, caption="", face_box=None) -> None
```

Four steps, in order:

**1. Crop to 16:9.** The raw frame might not be exactly 16:9 depending on the source video's aspect ratio, so `_crop_box_for_aspect` computes a centered crop box for the longer dimension. Unlike a plain crop-and-forget, this function's *return value* (the crop box) is kept, because step 4 needs it to translate face coordinates into the final image.

**2. Resize to exactly 1280×720** — YouTube's standard thumbnail resolution — using `Image.LANCZOS` for a high-quality resample. The resize scale factor is also kept for the same reason as the crop box.

**3. Color-and-contrast "pop"** (`_boost_colors`) — a real designed thumbnail is punchier than a raw video frame, so this always applies:
```python
img = ImageOps.autocontrast(img, cutoff=1)       # stretch the tonal range
img = ImageEnhance.Contrast(img).enhance(1.2)    # +20% contrast
img = ImageEnhance.Color(img).enhance(1.35)      # +35% saturation
img = ImageEnhance.Sharpness(img).enhance(1.15)  # +15% sharpness
```

**4. Draw the caption, avoiding the face** (`_draw_caption`) — this is the part that changed the most, in response to real feedback that early versions of this pipeline sometimes drew the caption right over the subject's face.

The original design asked *Gemini* to guess whether the caption should go "top" or "bottom," based on an 800px-wide downsized copy of the image. That was the wrong tool for the job: Gemini has no pixel-precise idea where the face actually is, so its guess was sometimes wrong. Meanwhile, `extract.py`'s own face detector (§4.3) had already computed the **exact bounding box** of the face on the full-resolution frame — that data just wasn't being used for placement. The fix was to stop asking the LLM to guess something the deterministic CV code already knew precisely:

- **`_map_face_box`** takes the face's bounding box (computed back in Stage ③, in the *original, uncropped* frame's pixel coordinates) and translates it through the exact same crop-offset and resize-scale used in steps 1–2, so it lines up correctly with the final 1280×720 canvas.
- **`_text_band`** then picks whichever of the top or bottom strip has more clear vertical space above/below the mapped face box (with a small safety margin), and returns that strip's boundaries. If no face was detected at all (e.g. a screen-share or gameplay video), it just defaults to a full-height bottom band — the classic thumbnail-caption position.
- The caption's **maximum font size is capped by how much room is actually available in that strip**, so a very close-up face (which leaves little clear space) automatically gets a smaller caption instead of overlapping anyway.
- A **font-fitting loop** (`_fit_caption`) starts at that capped size and shrinks it in steps until the text wraps to **at most 2 lines** that fit within the image width (minus margins), using a simple greedy word-wrap (`_wrap_text`).
- The font is **Impact** (the classic bold, condensed thumbnail/meme typeface) if present on the system, falling back to **Arial Bold**, then PIL's built-in default font as a last resort.
- **A translucent rounded dark panel is drawn behind the text** (`ImageDraw.rounded_rectangle` on a separate `RGBA` overlay, alpha-composited onto the frame) before the text itself is drawn. This was the second half of the feedback fix — plain outlined text floating directly on a busy photo background reads as flat/unpolished; a soft dark panel behind it is what most real thumbnail templates do to guarantee contrast and make the text read as a deliberate design element rather than an afterthought.
- Finally, the caption is rendered on top of the panel with a **thick black stroke outline around white fill** (`stroke_width`, `stroke_fill="black"`).

The result is saved as a JPEG at quality 90.

> **Why not just have Gemini return a bounding box instead of "top"/"bottom"?** We could have — but the face detector already computed a pixel-perfect box on the full-resolution frame, for free, as part of scoring. Asking an LLM to re-derive that same information (less precisely, from a downsized copy) would be redundant. This is the same "right tool for the job" principle from [§2](#2-why-two-layers-the-core-design-decision): once you already have exact structured data from deterministic code, don't ask an LLM to estimate it again.

---

## 5. Two Ways to Run It: CLI vs. Web App

Both entry points call the **exact same pipeline functions** (`download_video`, `get_shortlist`, the optional `transcribe`, `pick_best_frame`, `make_thumbnail`) — there's no duplicated pipeline logic, just two different ways of driving it.

### 5.1 CLI (`main.py`)

```powershell
python main.py "<youtube-url>" -o thumbnail.jpg
python main.py "<youtube-url>" --shortlist-only   # stop after Stage 1, no API call
```

`main.py`'s `run()` function is a thin, linear script: call each stage in order, print progress and the chosen candidate's reasoning/caption to the console, then save the output. The `--shortlist-only` flag exists specifically so you can iterate on the free, local part of the pipeline (download/sample/score) without spending Gemini API quota on every test run.

### 5.2 Web App (`app.py` + `templates/index.html`)

```powershell
python app.py    # serves http://127.0.0.1:5000
```

`app.py` is a small **Flask** application with two routes:

| Route | Method | Job |
|---|---|---|
| `/` | GET | Renders `templates/index.html` — the URL input form |
| `/generate` | POST | Runs the full pipeline synchronously, then re-renders the same page with a result (or an error) |

**Why synchronous?** The pipeline takes 30–90 seconds. For a miniproject, the simplest thing that works is to just let the browser's POST request sit and wait — no job queue, no polling, no WebSockets. The page shows a "Working..." message (a few lines of plain JavaScript on form submit) so it's not confusing while it waits.

**Error handling is deliberately visible, not silent.** `/generate` catches exceptions from any stage (missing API key, download failure, Gemini overloaded, no usable frames) and re-renders the page with a friendly error message instead of crashing the server or returning a blank 500 page.

**Output storage.** Each generated thumbnail is saved to `static/outputs/<video_id>.jpg`, which Flask serves directly as a static file — the `<img>` tag in the result just points at that URL (with a `?v=<timestamp>` cache-busting query param, so regenerating the same video doesn't show a stale cached image).

---

## 6. Configuration Reference (`config.py`)

Every tunable constant lives in one place, with the reasoning behind each documented here rather than scattered as inline comments:

| Constant | Value | Why |
|---|---|---|
| `SAMPLE_INTERVAL_SECONDS` | 2 | Base frame-sampling interval before the cap kicks in |
| `MAX_SAMPLED_FRAMES` | 150 | Hard ceiling regardless of video length — bounds scoring time |
| `SKIP_INTRO_OUTRO_FRACTION` | 0.03 | Skip first/last 3% of the video (intros/outros/end-cards) |
| `MAX_DOWNLOAD_HEIGHT` | 720 | No point downloading higher-res source than the 720p output |
| `MAX_VIDEO_DURATION_SECONDS` | 7200 (2h) | Hard cap checked *before* downloading — a long video means gigabytes to download and hours of ffmpeg decoding regardless of how few frames get kept. See [§7.6](#76-a-10-hour-video-silently-ran-for-over-an-hour-with-no-output) |
| `SHORTLIST_SIZE` | 8 | How many top-scoring frames get shown to Gemini |
| `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` | 1280 / 720 | YouTube's standard thumbnail resolution |
| `MIN_LAPLACIAN_VAR` | 20 | Hard blur floor — frames below this are rejected, not just scored low |
| `WEIGHT_SHARPNESS` / `WEIGHT_FACE` / `WEIGHT_EXPOSURE` | 0.4 / 0.4 / 0.2 | Heuristic scoring weights — tune these by eye against real videos |
| `GEMINI_MODEL` | `"gemini-flash-lite-latest"` | An *alias*, not a pinned version — moves with Google's current lineup instead of going stale. Specifically the "Lite" tier, not plain `"gemini-flash-latest"`, because Lite has a separate (and much less easily exhausted) free-tier quota bucket — see [§7.5](#75-gemini-flash-latest-hit-its-daily-free-tier-quota-wall) |
| `GEMINI_IMAGE_MAX_WIDTH` | 800 | Images are downsized to this before upload, to control vision token cost |
| `FACE_MODEL_PATH` / `FACE_MODEL_URL` | — | Where the YuNet face model lives locally, and where to auto-download it from if missing |
| `TRANSCRIBE_MODEL` | `"whisper-1"` | OpenAI's transcription model — only used if `OPENAI_API_KEY` is set (§4.3.5) |
| `TRANSCRIBE_WINDOW_SECONDS` | 6.0 | How wide a window around a candidate's timestamp counts as "speech near this moment" |

---

## 7. Real Bugs We Hit and How We Fixed Them

These aren't hypothetical — they're the actual failures we ran into getting this pipeline working, kept here because they're genuinely instructive about building on top of fast-moving libraries/APIs.

### 7.1 OpenCV 5.0 deleted the classic face detector

Every OpenCV tutorial you'll find uses:
```python
cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```
On the version installed here (**OpenCV 5.0**), this fails: `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`. The old Haar-cascade API — and the bundled `.xml` model files that shipped inside the `opencv-python` package — were removed entirely in OpenCV 5.0, replaced by a DNN-based detector called **`FaceDetectorYN`** (YuNet).

The fix (`extract.py`, `_ensure_face_model` / `_build_face_detector`): download a small (~230KB) YuNet ONNX model file on first use from the OpenCV Zoo GitHub repo, then build the detector with:
```python
cv2.FaceDetectorYN_create(model_path, "", (frame_width, frame_height))
```
and call `.detect(frame)` instead of `.detectMultiScale(gray)`. If the model can't be downloaded for any reason, `_build_face_detector` returns `None` and every frame just scores `has_face=False` — the pipeline degrades gracefully instead of crashing.

**Lesson:** don't trust a library's most commonly-tutorialized API to still exist in whatever version actually gets installed — check what's real in the installed package before writing code against it.

### 7.2 `gemini-2.5-flash` was already deprecated for new API keys

The model name that was the obvious, well-documented choice returned:
```
404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to new users.
```
even though it still showed up in `client.models.list()` — meaning it still exists for *some* accounts, just not new ones. We tested several models directly and found `gemini-2.0-flash` hit a `429 RESOURCE_EXHAUSTED` (quota), while **`gemini-flash-latest`** — an alias, not a version-pinned name — worked immediately.

**Lesson:** when a fast-moving API offers an alias like `-latest`, prefer it over a pinned version for exactly this reason: it moves with the provider's own lineup instead of quietly aging out.

### 7.3 Gemini's free tier returns transient 503s

During testing, back-to-back requests to the same model returned:
```
503 UNAVAILABLE: This model is currently experiencing high demand.
```
This is expected free-tier behavior, not a bug in our code — but a pipeline that just crashes on the first transient overload is a bad experience. We added a small retry loop in `agent.py` (up to 3 attempts, 3s/6s backoff) that only catches `genai_errors.ServerError` (5xx-class errors) — not 4xx errors like a bad API key, which should fail immediately rather than retry.

### 7.4 A slow/unstable network stalled the ffmpeg install

Not a code bug, but worth recording: the initial `winget install` of ffmpeg stalled at 0 bytes downloaded for several minutes. Switching to a direct `curl` download of the smaller "essentials" build (instead of the 253MB "full" build `winget` was fetching) got it done in a couple of minutes once the connection recovered. If you ever hit this yourself: check whether the installer is actually making progress before assuming it's just slow.

### 7.5 `gemini-flash-latest` hit its daily free-tier quota wall

After enough test runs, requests started failing with:
```
429 RESOURCE_EXHAUSTED: Quota exceeded for metric: generate_content_free_tier_requests,
limit: 20, model: gemini-3.5-flash ... quotaId: 'GenerateRequestsPerDayPerModel-FreeTier'
```
This is a **different failure mode** from the transient 503s in §7.3 — it's a hard daily cap (20 requests/day for whichever model `"gemini-flash-latest"` currently resolves to), not something a short retry-with-backoff fixes. The response even includes a `retryDelay` of a few seconds, which is misleading for a *per-day* quota — retrying immediately just fails again.

Two things were wrong here, and both got fixed:

1. **The raw Google error dict was being shown directly in the UI** — accurate, but unhelpful (a wall of `{'error': {'code': 429, ...`). `agent.py` now catches `genai_errors.ClientError` specifically, checks whether the quota violation says `"PerDay"` (checking `str(e.details)`), and raises a plain-English `RuntimeError` explaining what actually happened and what to do about it — wait for the reset, switch models, or enable billing.
2. **The model itself was too easily exhausted.** Testing showed that different Gemini model tiers have *separate* quota buckets: `"gemini-flash-latest"` and `"gemini-2.0-flash"` were both exhausted, but `"gemini-flash-lite-latest"` (the "Lite" tier — smaller/faster, still fully capable of vision + function calling for this use case) still had quota available. `config.GEMINI_MODEL` was switched to it. This is the same "alias, not a pinned version" reasoning as §7.2, just one tier down — a Lite alias moves with Google's lineup too, and its quota resets independently of the regular Flash tier's.

**Lesson:** a 429 isn't automatically "wait and retry" — read the actual quota metadata (`quotaId`, `quotaDimensions.model`) to tell a short rate limit apart from a daily wall, and remember that different model tiers on the same API key can have completely independent quota buckets.

### 7.6 A 10-hour video silently ran for over an hour with no output

A user pasted a link to a **10-hour course video**. The web app just sat on "Working..." indefinitely — no error, no progress, nothing. Digging in with `Get-CimInstance Win32_Process` to see what the Python process was actually doing turned up an `ffmpeg` child process that had been running for over half an hour:
```
ffmpeg -ss 1080.15 -i source.mp4 -t 33844.7 -vf fps=0.00417 ... frame_%04d.jpg
```
`-t 33844.7` is a **9.4-hour** trim window. Nothing was actually broken — `MAX_SAMPLED_FRAMES` (§4.2) correctly widened the sampling interval so the frame *count* stayed capped at 150, exactly as designed. The problem was something the design never accounted for: **there was no limit on video length in the first place.** A 10-hour video meant downloading a ~1.5GB file and then having ffmpeg decode through the entire 9+ hour span (sparse output sampling doesn't mean sparse decoding — most codecs still have to decode through the frames *between* the ones being kept) — real wall-clock hours, not the 30–90 seconds the UI promised.

There was a second, compounding problem: Flask's dev server runs **single-threaded by default**, so once the long request started, the *entire server* — including the homepage — stopped responding to anything else. From the user's side, this looked like the app had frozen completely, not just one slow request.

Two fixes, both in the "check assumptions the design implicitly relied on but never verified" category:

1. **`download.py` now probes the video's duration *before* downloading anything** (`_probe_duration`, a metadata-only `yt_dlp.extract_info(url, download=False)` call — a couple seconds, no download). If it's longer than `config.MAX_VIDEO_DURATION_SECONDS` (2 hours, tunable), it raises a clear error immediately instead of silently attempting a doomed multi-hour job. Live streams (which report no fixed duration) are rejected the same way, for the same reason — there's no sensible "finish downloading" point for an unbounded stream.
2. **`app.py` now passes `threaded=True` to `app.run()`**, so one slow `/generate` request no longer blocks every other request (including the homepage) from being served.

**Lesson:** "cap the frame count" and "cap the video length" are two different constraints — capping one doesn't cap the other, and the download + decode cost scales with the *source* video's length regardless of how few frames you keep. Any pipeline stage whose cost scales with an input the user controls (video length, file size, list length) needs its own explicit ceiling, checked as early and as cheaply as possible — ideally before the expensive part (the download) even starts, not after.

---

## 8. What's Deliberately Not Here (Stretch Goals)

These were left out on purpose to keep the project scoped and reliable, not because they'd be hard to bolt on:

- **Subject cutout / background replacement** (MrBeast-style pop-out): would need a background-removal model (e.g. `rembg`, ~175MB), is slower, and can produce rough edges on some frames.
- **Avoiding *all* on-screen graphics, not just faces**: text placement now precisely avoids the detected face (§4.5), but it doesn't know about other important on-screen elements — a lower-third logo, an existing caption baked into the source video, etc. It can still end up overlapping something that isn't a face. A fuller fix would run a general "busy region" detector (e.g. flag areas with high edge density or existing text) rather than just faces.
- **Multiple faces**: if a frame has more than one face, only the largest one is currently protected from overlap — a two-person interview frame could still get a caption over the smaller face.
- **Full agentic orchestration**: right now only the frame-picking/captioning step is a Gemini tool call — download/sampling/composing are plain deterministic function calls in `main.py`/`app.py`, not something Gemini decides to invoke itself. That's intentional: there's no judgment call in "download the video" or "where exactly is the face" (we already know that precisely), so there's no reason to route those through an LLM.
- **A job queue for the web app**: `/generate` blocks synchronously for the full 30–90 second pipeline run. Fine for local/single-user use; a real multi-user deployment would want a background task queue instead.
- **Transcript-aware frame selection, not just captioning**: `transcribe.py` (§4.3.5) currently only feeds nearby speech into the *caption wording* — Gemini isn't told to prefer a frame specifically because of a strong verbal claim being made at that exact instant. A fuller version would fold "how compelling is the speech right now" into frame scoring too, not just caption writing.
