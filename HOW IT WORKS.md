# How It Works

This document explains, in depth, how Thumbsmith turns a video into a designed thumbnail — what each file does, why it's built the way it is, and the real bugs we hit and fixed along the way.

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Why Three Layers? (The Core Design Decision)](#2-why-three-layers-the-core-design-decision)
3. [Project File Map](#3-project-file-map)
4. [Stage-by-Stage Walkthrough](#4-stage-by-stage-walkthrough)
   - [4.1 Getting the video (`download.py`)](#41-getting-the-video-downloadpy)
   - [4.2 Frame Sampling (`extract.py` — part 1)](#42-frame-sampling-extractpy--part-1)
   - [4.3 Heuristic Scoring (`extract.py` — part 2)](#43-heuristic-scoring-extractpy--part-2)
   - [4.3.5 (Optional) Speech Transcription (`transcribe.py`)](#435-optional-speech-transcription-transcribepy)
   - [4.4 Talking to kie.ai (`kie.py`)](#44-talking-to-kieai-kiepy)
   - [4.5 The Art Director (`director.py`)](#45-the-art-director-directorpy)
   - [4.6 The Render (`generate.py`)](#46-the-render-generatepy)
   - [4.7 Composing the Headline (`compose.py`)](#47-composing-the-headline-composepy)
5. [Two Ways to Run It: CLI vs. Web App](#5-two-ways-to-run-it-cli-vs-web-app)
   - [5.1 CLI (`main.py`)](#51-cli-mainpy)
   - [5.2 Web App (`app.py` + `templates/index.html`)](#52-web-app-apppy--templatesindexhtml)
6. [Configuration Reference (`config.py`)](#6-configuration-reference-configpy)
7. [Real Bugs We Hit and How We Fixed Them](#7-real-bugs-we-hit-and-how-we-fixed-them)
8. [What's Deliberately Not Here (Stretch Goals)](#8-whats-deliberately-not-here-stretch-goals)

---

## 1. The Big Picture

**Input:** a YouTube URL, or a video file you upload.
**Output:** a 1280×720 thumbnail JPEG — the strongest frame, re-rendered as a designed image, with a headline placed where it will actually be read.

```
YouTube URL ──┐
              ├──► ① Get the video                  — download.py
video file ───┘         │
                        ▼
                 ② Sample ~100-150 frames (ffmpeg)  — extract.py
                        │
                        ▼
                 ③ Score + de-duplicate             — extract.py
                    (sharpness, face, exposure,
                     colour, composition)
                        │
                        ▼  (top 8 candidates only)
                 ③.5 (optional) Transcribe audio    — transcribe.py
                     with Whisper, if OPENAI_API_KEY is set
                        │
                        ▼
                 ④ Art director → ThumbnailPlan     — director.py
                    (frame, headline, layout,
                     accent colour, art direction)
                        │
                        ▼
                 ⑤ GPT Image 2 re-renders the       — generate.py
                    chosen frame, image-to-image
                        │
                        ▼
                 ⑥ Draw the headline                — compose.py
                        │
                        ▼
                  thumbnail.jpg
```

Steps ①–③ and ⑥ are **plain deterministic code** — no AI, no API calls, free, fast, reproducible. Steps ④ and ⑤ are the model stages, and both run on [kie.ai](https://kie.ai) under a single `KIE_API_KEY`.

`pipeline.py` holds this whole sequence in one function, `generate_thumbnail(source, output_path)`. The CLI and the web app are both thin wrappers around it.

---

## 2. Why Three Layers? (The Core Design Decision)

The first question this project raises is: **do you need a model at all, or can you just write scoring math?**

The answer: **you need three different things, and only two of them are a model.**

| Layer | What it does | Technology | Cost |
|---|---|---|---|
| **1 — pre-filter** | Get the video, sample ~100-150 frames, score each on sharpness/face/exposure/colour/composition, de-duplicate, keep the top 8 | Python + OpenCV + ffmpeg | Free, no API calls |
| **2 — judgment** | Look at those 8 and decide which one earns the click, what to write on it, and how it should be laid out | kie.ai chat model (vision) | One request |
| **3 — craft** | Turn that plan into a picture that looks designed rather than screenshotted | GPT Image 2 on kie.ai, then local typography | One render |

**Why not show the model all 150 frames?** Most are near-duplicates or obviously bad — blurry, mid-blink, transition frames. Spending tokens on them buys nothing, and a long image list makes the model's attention worse, not better.

**Why not skip the model and use the heuristic score directly?** Because the heuristic has no idea what's *on screen*. Every frame in our test video scores almost identically on sharpness/face/exposure — they're all clips of the same well-lit talking-head shot, and the heuristic literally cannot tell them apart. It also has no opinion whatsoever about what to write on the image, which is half the job.

**Why is there a render step at all — why not just put text on the frame?** That was the old design, and its ceiling is low. A raw video still is lit for video, not for a 210×118 tile in a grid: flat contrast, a busy background competing with the subject, no separation. GPT Image 2 fixes the things a colour-boost filter cannot — relighting the subject, pushing the background back, clearing clutter out of the space the headline needs.

**Why image-to-image and not text-to-image?** Because a thumbnail's job is to represent *this* video. A prompt-generated picture invents a stranger in a scene that never happens. That's both dishonest and, in practice, a worse click — viewers bounce when the thumbnail doesn't match the first ten seconds, and that hurts the video more than a weak thumbnail does.

**Why is the headline drawn locally instead of by the image model?** GPT Image 2 can render text, and `config.LET_MODEL_RENDER_TEXT` will let it. It's off by default because local text is the only way to guarantee three things at once: the exact words with no spelling drift, a size that survives being shrunk to thumbnail scale, and a position that is provably clear of the face and of YouTube's duration badge. Those are geometry problems with exact answers, and §2's principle applies — don't ask a model to estimate something deterministic code already knows precisely.

---

## 3. Project File Map

```
thumbnail_picker/
    __init__.py         (empty — makes this a Python package)
    config.py           All tunable constants live here
    download.py         YouTube URL or local file -> one VideoSource
    extract.py          video -> sampled frames -> scored, de-duplicated shortlist
    transcribe.py       (optional) video -> speech transcript, via OpenAI Whisper
    kie.py              the only module that talks to kie.ai
    director.py         shortlist (+ transcript) -> ThumbnailPlan
    generate.py         chosen frame + plan -> GPT Image 2 render
    compose.py          rendered image + plan -> final 1280x720 JPEG
    pipeline.py         the whole run, start to finish
main.py                 CLI entrypoint
app.py                  Flask web app
templates/
    index.html          Web UI: link tab + upload tab, and the result display
static/
    outputs/            Generated thumbnails served by the web app
models/
    face_detection_yunet_2023mar.onnx   Face-detection model (auto-downloaded)
work/                   Scratch: videos, sampled frames, and renders per video ID
requirements.txt        pip dependencies
.env.example            Every environment variable, documented
```

Nothing here is a black box — every stage is a plain function you can call directly and inspect, which is why each one could be tested independently.

---

## 4. Stage-by-Stage Walkthrough

### 4.1 Getting the video (`download.py`)

```python
def download_video(url: str) -> VideoSource
def ingest_local_video(path: str, display_name=None, move=False) -> VideoSource
```

Two entry points, one return type. Everything downstream reads `VideoSource` (path, id, title, duration, origin) and never needs to know which one produced it.

**From YouTube.** Uses **`yt-dlp`** via its Python API — not the command line — so failures raise catchable exceptions instead of printing to a terminal. The format string caps the download at 720p:

```python
"format": "bestvideo[height<=720]+bestaudio/best[height<=720]"
```

There's no reason to pull a 4K source when the output is 1280×720. Video and audio are downloaded separately and merged into one `.mp4`, which is why **ffmpeg is a required external dependency**.

**From a file.** `ingest_local_video` validates the extension against `ALLOWED_UPLOAD_EXTENSIONS`, checks the size against `MAX_UPLOAD_BYTES`, and reads the duration with **ffprobe** — the local equivalent of the metadata probe the YouTube path does, and the same 2-hour ceiling applies (see [§7.6](#76-a-10-hour-video-silently-ran-for-over-an-hour-with-no-output)).

The file is placed in `work/<id>/` rather than read in place, so the pipeline writes frames next to the video exactly as it does for a download. It's **copied** by default, leaving the user's own file untouched; the web uploader passes `move=True` because its source is a throwaway temp file and copying half a gigabyte twice is pure waste.

The id is derived from the filename and size (`upload-my-clip-a1b2c3d4e5`), so re-uploading the same file reuses its work directory instead of filling the disk with duplicates.

### 4.2 Frame Sampling (`extract.py` — part 1)

```python
def sample_frames(video_path, duration, frames_dir) -> list[(path, timestamp)]
```

Calls `ffmpeg` as a subprocess to pull frames at a fixed interval:

```
ffmpeg -ss <skip> -i source.mp4 -t <trimmed_duration> -vf "fps=1/interval" frames/frame_%04d.jpg
```

**1. The interval adapts to video length**, capped so no video produces more than `MAX_SAMPLED_FRAMES` (150) frames:
```python
def _sample_interval(duration):
    if natural_count <= max_count:
        return SAMPLE_INTERVAL_SECONDS   # normally 1 frame every 2 seconds
    return duration / max_count          # widen the gap for very long videos
```
A 10-minute video samples every 2 seconds (~300 frames → capped at 150, so really every 4). A 2-hour video samples roughly every 48 seconds. This bounds scoring time regardless of input.

**2. The first/last 3%** (`SKIP_INTRO_OUTRO_FRACTION`) is skipped — intros, outros, and end-cards are poor thumbnail material.

**3. Stale frames are deleted first.** Re-running on a video whose work directory already exists used to leave the previous run's `frame_*.jpg` files in place, so a shorter second run would mix old frames into the new shortlist with wrong timestamps attached.

### 4.3 Heuristic Scoring (`extract.py` — part 2)

```python
def score_frames(sampled) -> list[Candidate]
def diversify(candidates, limit) -> list[Candidate]
def get_shortlist(video_path, duration, frames_dir) -> list[Candidate]
```

Five signals per frame, combined into one score.

**Sharpness** — Laplacian variance of the greyscale image. A Laplacian highlights edges; a sharp frame has strong edges (high variance), a blurry or motion-smeared one has muddy edges.
```python
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
sharpness_score = min(lap_var / 300.0, 1.0)
```
Frames below `MIN_LAPLACIAN_VAR` (20) are rejected outright — a hard floor, not a low score.

**Face** — YuNet records the largest face's exact pixel box (see [§7.1](#71-opencv-50-deleted-the-classic-face-detector)). The score is *not* a boolean. A face that fills 1% of the frame is a person standing somewhere in a wide shot, which is dead weight at thumbnail size; `_face_scores` ramps quality up to `IDEAL_FACE_AREA_FRACTION` (12%) and then tapers back down, because an extreme close-up crops out context and leaves nowhere for text.

**Exposure** — from the greyscale histogram: what fraction of pixels are dead black or blown white, and how close the mean is to a well-exposed 125/255.

**Colourfulness** — the Hasler–Süsstrunk metric. Flat, grey, washed-out frames vanish in a YouTube grid no matter how sharp they are, and this is the cheapest reliable proxy for "does this frame have any punch".

**Composition** — where the face sits. Peaks at the rule-of-thirds verticals (⅓ or ⅔ across) and around 42% down, worst dead-centre or jammed against an edge. An off-centre subject is what leaves usable negative space for the headline, so this signal directly serves the layout stage.

```python
total = 0.25 * sharpness + 0.35 * face_quality + 0.15 * exposure
      + 0.15 * colourfulness + 0.10 * composition
```

**De-duplication is the step that matters most.** Sorting by score and taking the top 8 sounds right and is quietly useless: in a talking-head video the eight highest-scoring frames are eight copies of the same shot, three seconds apart, and the art director has no real choice to make. `diversify()` walks candidates best-first and keeps one only if it is both **visually** different from everything kept so far (64-bit DCT perceptual hash, Hamming distance > `DEDUPE_HASH_DISTANCE`) and **temporally** separated (> `MIN_CANDIDATE_GAP_SECONDS`). If a short or very static video can't fill the shortlist that way, it backfills on score rather than returning two candidates.

### 4.3.5 (Optional) Speech Transcription (`transcribe.py`)

```python
def transcribe(video_path: str) -> list[Segment]      # Segment = (start, end, text)
def text_near(segments, timestamp, window=6.0) -> str
```

**Entirely optional** — it runs only if `OPENAI_API_KEY` is set. Without it, `is_available()` returns `False`, callers skip the stage, and the pipeline behaves as if the file didn't exist. This matters because transcription is genuinely cheap (~$0.006/minute) but it *is* real billed usage, so it shouldn't be a silent requirement.

Why it exists: the art director only ever *sees* the frames. It has no idea what the speaker is saying. A headline grounded in an actual spoken claim ("I saved $3,459 doing this") is a stronger hook than one guessed from pixels.

**How it works:**
1. **Extract audio only** (`-vn -acodec libmp3lame -b:a 64k -ar 16000 -ac 1`). The transcription endpoint caps uploads at 25MB; a mono 16kHz 64kbps MP3 is roughly 0.5MB per minute, so even a long video fits, where the merged video file would not.
2. **Transcribe with Whisper** (`response_format="verbose_json"`), which returns **timestamped segments** — what's needed to answer "what was said around second 493?" rather than "what was said somewhere in this video?"
3. **`text_near`** concatenates segments overlapping a ±6s window around a candidate's timestamp.
4. **Every failure degrades to "no transcript", never a crash.** Missing key, missing package, network failure, ffmpeg failure, malformed response — all caught, all return `[]`.

The director labels each candidate with its nearby speech, and the brief tells it to anchor the headline in something actually said.

### 4.4 Talking to kie.ai (`kie.py`)

Every network call to kie.ai lives in this one module. Nothing else in the codebase knows the platform exists, which is what makes swapping models a config change rather than a refactor.

Three shapes of call:

**`upload_image(path, max_width)`** — every other kie.ai endpoint takes image *URLs*, not bytes, so frames are downsized, JPEG-encoded, base64'd, and POSTed to the file-upload endpoint, which returns a public `downloadUrl` that lives for 24 hours.

**`chat(messages, model)`** — an OpenAI-compatible chat completion, used for the vision stage. Synchronous; the answer is in `choices[0].message.content`. Content that comes back as typed parts rather than a bare string is flattened, since deployments differ on that.

**`run_image_task(model, input)`** — the asynchronous job flow the image models use. POST to `createTask`, get a `taskId`, then poll `recordInfo` until `state` reaches `success` (result URLs live in the `resultJson` string) or `fail`. Polling runs every 3s up to `KIE_POLL_TIMEOUT_SECONDS`.

**Error handling is the bulk of this file, on purpose.** kie.ai returns its real status in the JSON body rather than only the HTTP code, so `_explain_http_error` reads the body first and turns the common cases into sentences a human can act on — a rejected key names the environment variable, an empty balance links to billing, a rate limit says to wait. `_post` retries transport errors and 5xx with backoff, and never retries a 4xx, because a bad key does not become a good key on attempt three.

### 4.5 The Art Director (`director.py`)

```python
def direct(shortlist, frame_urls, transcript_segments=None, title="") -> ThumbnailPlan
```

This stage returns a **plan, not a pick**:

```python
@dataclass
class ThumbnailPlan:
    candidate_id: str      # which frame
    reason: str            # why it wins
    headline: str          # 2-4 words, ALL CAPS, <= 28 chars
    accent_word: str       # the one word to colour, or ""
    subject_side: str      # left | right | center
    accent_color: str      # #RRGGBB
    scene_direction: str   # art direction for the image model
```

**Why one structured plan instead of separate calls.** The render needs to know where to leave negative space; the compositor needs to know where the text goes; both need to agree. Deriving them from one answer is what keeps the caption, the empty half of the picture, and the text placement consistent. `plan.text_side` is a property, not a field — it's just the opposite of `subject_side`, and computing it beats trusting the model to keep two fields in sync.

**The brief encodes actual thumbnail craft**, not a vague "pick a good frame": a large readable face with a *specific* emotion beats everything; reject mid-blink and mid-motion; prefer a frame with room around the subject; write a hook, never a description; short words win because the thing is judged at 210×118 in under a second.

**Validation is not optional.** Model JSON is parsed defensively and every field is repaired or rejected:
- `_extract_json` strips markdown fences and, failing that, slices from the first `{` to the last `}` — models pad JSON with prose.
- `candidate_id` must be in the shortlist, or the plan is rejected outright.
- The headline is uppercased, stripped of trailing punctuation, and trimmed **on a word boundary** if over-length — a clipped word reads as a bug.
- `accent_word` is kept only if it really is one of the headline's words, otherwise the compositor would highlight nothing.
- `subject_side` falls back to `center`, `accent_color` to the configured default, if either is malformed.

On a validation failure the model is shown its own bad output and asked once more. One retry only — a model that can't produce valid JSON twice won't on the third try, and each attempt costs credits.

### 4.6 The Render (`generate.py`)

```python
def render(frame_path, plan, dest_path) -> str
def render_or_fallback(frame_path, plan, dest_path) -> tuple[str, str | None]
```

The chosen frame is uploaded at full width (the director's copies were downsized to 800px for token cost; this one *is* the input the render is built from) and sent to `gpt-image-2-image-to-image` with a prompt assembled from three parts:

1. **A fixed base direction** — keep the same person, same face, same clothing, same expression; do not restyle into an illustration. Then: grade for contrast and saturation, key-light the face, rim-light the edge, push the background darker and blurred, remove clutter, keep skin natural.
2. **A negative-space clause** keyed off `subject_side` — "keep the subject filling the RIGHT half, the LEFT half must stay visually calm and uncluttered". This is the instruction that makes the layout work.
3. **The director's `scene_direction`**, which is the only shot-specific part.

Plus a hard **no-text rule**, because any letters the model draws would collide with the headline composited in the next stage.

**Failure is survivable.** `render_or_fallback` catches `KieError` and returns the original frame plus a warning instead of failing the run. A user who is out of credits or hits an outage still gets a graded, captioned thumbnail — worse than the rendered one, but a result. The warning is surfaced in the UI rather than swallowed, so nobody is misled about which they got.

### 4.7 Composing the Headline (`compose.py`)

```python
def make_thumbnail(image_path, output_path, headline="", accent_word="",
                   accent_color=..., preferred_side="bottom", grade_strength=1.0) -> dict
```

**1. Crop to 16:9, focused on the face.** `_crop_box_for_aspect` takes an optional focal point, so cropping a tall frame doesn't slice the subject's head off.

**2. Grade — but only as much as is needed.** `grade_strength` is 1.0 for a raw still and 0.25 for a GPT Image 2 render. The render already arrives graded; running the full boost over it again crushes the highlights.

**3. Re-detect the face on the *final* canvas.** This is the non-obvious one. The crop, the resize, and above all the render itself all move the subject, so the plan's idea of which half is empty can be stale by the time the picture comes back. Placement has to answer to where the face is *now*, not where it was in the source frame.

**4. Choose a zone.** Four candidate zones — left column, right column, bottom band, top band — each already inside the safe margin and clear of the **duration badge** YouTube stamps over the bottom-right corner.

Zones are judged by **clearance, not overlap**: the tallest contiguous strip the face does not cut through. This distinction is the whole ball game. Measuring covered *area* lets a small, dead-centre face look harmless in a wide top band — right up until the headline is painted straight across it. Measuring the surviving strip asks the real question: is there room for the text? The director's chosen side is kept whenever it clears `MIN_CLEARANCE_FRACTION`, and only overridden when the face genuinely leaves no room, because second-guessing the plan on a technicality throws away the negative space the render was built around.

**5. Place the block within the zone.** `_block_top` centres the text vertically, then slides it above or below the face if it intrudes. This is what lets a tall column still work for a subject whose head sits high in the frame, instead of bouncing the headline to a different side entirely.

**6. Fit the type.** Shrink from `0.23 × height` until the wrapped lines fit — at most 3 lines in a column, 2 in a band. The available width subtracts **twice the stroke width**, because `textlength` measures glyphs only and the outline adds thickness on every side; without that allowance the headline overhangs the exact margin YouTube crops on some surfaces.

**7. Draw it.** A dark gradient **scrim** fades in from the headline's side, solid at the edge and gone by 70% across, so type always has something to sit on while the subject stays untouched. Its strength adapts to how bright and busy the zone measured. Then a Gaussian-blurred **drop shadow** on its own layer, then the text itself — white with a heavy black stroke, drawn word by word so the **accent word** can take the plan's colour.

Saved as JPEG at quality 92. Returns `{"text_side", "face_detected"}` so the caller can report what actually happened.

---

## 5. Two Ways to Run It: CLI vs. Web App

Both call `pipeline.generate_thumbnail`. They differ only in how they get a video and how they report progress — there is no duplicated pipeline logic.

### 5.1 CLI (`main.py`)

```powershell
python main.py "<youtube-url>" -o thumbnail.jpg
python main.py --file "C:\clips\video.mp4" -o thumbnail.jpg
python main.py "<youtube-url>" --shortlist-only   # stop after Stage 1, no API calls
```

The URL and `--file` are a mutually exclusive required group, so argparse rejects both-or-neither itself. `--shortlist-only` exists so you can iterate on the free, local part of the pipeline without spending credits on every test run — it also prints the full per-frame score breakdown, which is how the scoring weights get tuned.

### 5.2 Web App (`app.py` + `templates/index.html`)

```powershell
python app.py    # serves http://127.0.0.1:5000
```

| Route | Method | Job |
|---|---|---|
| `/` | GET | Renders the form — link tab and upload tab |
| `/generate` | POST | Runs the pipeline synchronously, re-renders the page with a result or an error |

**Two inputs, one form.** The tabs are cosmetic; both fields live in the same `multipart/form-data` form. On submit, JavaScript clears whichever field belongs to the *hidden* tab, so a stale URL left over from an earlier attempt can't quietly win over the file the user just dropped in. The server independently prefers the upload when both arrive, so the behaviour is defined even with JS off.

**Upload limits are enforced twice.** `MAX_CONTENT_LENGTH` makes Flask reject an oversized body before it's fully read, and a `RequestEntityTooLarge` handler turns that into the same friendly card as any other error rather than a raw 413 page. `ingest_local_video` re-checks size and extension, because the CLI path never goes through Flask at all.

**Why synchronous?** The pipeline takes 1–3 minutes. For a project this size, letting the POST sit and wait is the simplest thing that works — no queue, no polling, no WebSockets. A "Working..." message appears on submit so the wait isn't confusing.

**Errors are visible, not silent.** `/generate` catches `KieError` and `RuntimeError` and re-renders with a readable message. Warnings — a render that fell back, a transcript that came back empty, a missing face — are shown alongside a *successful* result, so a degraded run is never passed off as a clean one.

**Output storage.** Each thumbnail is saved to `static/outputs/<video_id>.jpg` and served as a static file, with a `?v=<timestamp>` cache-buster so regenerating the same video doesn't show a stale image.

---

## 6. Configuration Reference (`config.py`)

| Constant | Value | Why |
|---|---|---|
| `SAMPLE_INTERVAL_SECONDS` | 2 | Base sampling interval before the cap kicks in |
| `MAX_SAMPLED_FRAMES` | 150 | Hard ceiling regardless of video length — bounds scoring time |
| `SKIP_INTRO_OUTRO_FRACTION` | 0.03 | Skip first/last 3% (intros/outros/end-cards) |
| `MAX_DOWNLOAD_HEIGHT` | 720 | No point downloading higher-res than the output |
| `MAX_VIDEO_DURATION_SECONDS` | 7200 (2h) | Checked *before* downloading. See [§7.6](#76-a-10-hour-video-silently-ran-for-over-an-hour-with-no-output) |
| `MAX_UPLOAD_BYTES` | 512 MB | Upload ceiling, enforced by Flask and re-checked on ingest |
| `ALLOWED_UPLOAD_EXTENSIONS` | mp4/mov/mkv/webm/avi/m4v | What ffprobe and ffmpeg reliably handle |
| `SHORTLIST_SIZE` | 8 | How many frames reach the art director |
| `OUTPUT_WIDTH` / `OUTPUT_HEIGHT` | 1280 / 720 | YouTube's standard thumbnail resolution |
| `MIN_LAPLACIAN_VAR` | 20 | Hard blur floor — rejected, not just scored low |
| `WEIGHT_*` | 0.25 / 0.35 / 0.15 / 0.15 / 0.10 | Sharpness, face, exposure, colour, composition. Tune by eye against real videos |
| `IDEAL_FACE_AREA_FRACTION` | 0.12 | Face size that scores best; smaller is weak, much larger leaves no room for text |
| `DEDUPE_HASH_DISTANCE` | 6 | Perceptual-hash distance below which two frames are "the same picture" |
| `MIN_CANDIDATE_GAP_SECONDS` | 8.0 | Forces the shortlist to span the video instead of clustering |
| `KIE_CHAT_MODEL` | `"gpt-5-2"` | The art director. Override with the `KIE_CHAT_MODEL` env var |
| `KIE_IMAGE_MODEL` | `"gpt-image-2-image-to-image"` | The renderer. Override with `KIE_IMAGE_MODEL` |
| `KIE_IMAGE_RESOLUTION` | `"2K"` | Rendered above 1280×720 so the downscale to output size stays crisp |
| `KIE_POLL_TIMEOUT_SECONDS` | 420 | How long to wait on a render before giving up |
| `DIRECTOR_IMAGE_MAX_WIDTH` | 800 | Frames are downsized this far before upload, to control vision token cost |
| `RENDER_IMAGE_MAX_WIDTH` | 1280 | The chosen frame goes to the image model larger — it *is* the render input |
| `LET_MODEL_RENDER_TEXT` | `False` | Let GPT Image 2 paint the headline instead of compositing it. See [§2](#2-why-three-layers-the-core-design-decision) |
| `FALLBACK_TO_RAW_FRAME` | `True` | On render failure, still produce a thumbnail from the original frame |
| `SAFE_MARGIN_FRACTION` | 0.045 | Keeps type off the edges YouTube crops |
| `DURATION_BADGE_*_FRACTION` | 0.22 / 0.17 | The bottom-right corner YouTube covers with the runtime |
| `FONT_CANDIDATES` | — | First existing font wins; Windows/macOS/Linux paths, `THUMBNAIL_FONT_PATH` first |
| `FACE_MODEL_PATH` / `_URL` | — | Where YuNet lives locally, and where to auto-download it from |
| `TRANSCRIBE_MODEL` | `"whisper-1"` | Only used if `OPENAI_API_KEY` is set (§4.3.5) |
| `TRANSCRIBE_WINDOW_SECONDS` | 6.0 | How wide a window counts as "speech near this moment" |

---

## 7. Real Bugs We Hit and How We Fixed Them

These aren't hypothetical — they're the actual failures we ran into, kept here because they're genuinely instructive about building on top of fast-moving libraries and APIs.

### 7.1 OpenCV 5.0 deleted the classic face detector

Every OpenCV tutorial uses:
```python
cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```
On **OpenCV 5.0** this fails: `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`. The Haar-cascade API and the bundled `.xml` models were removed entirely, replaced by a DNN detector, **`FaceDetectorYN`** (YuNet).

The fix (`_ensure_face_model` / `_build_face_detector`): download the small (~230KB) YuNet ONNX model on first use from the OpenCV Zoo, then build the detector with `cv2.FaceDetectorYN_create(path, "", (w, h))` and call `.detect(frame)` instead of `.detectMultiScale(gray)`. If the model can't be downloaded, `_build_face_detector` returns `None`, every frame scores `has_face=False`, and the pipeline degrades instead of crashing.

**Lesson:** don't trust a library's most-tutorialized API to still exist in the version that actually gets installed.

### 7.2 The text placement was solving the wrong problem

The first version of the compositor asked the *model* whether the caption should go "top" or "bottom", based on an 800px downsized copy. It guessed wrong often enough to matter, drawing captions across faces. That got replaced with deterministic placement off the face detector's exact box — the right call, and the general principle in [§2](#2-why-three-layers-the-core-design-decision).

But the replacement had a subtler bug that survived much longer: it scored each zone by **how much of the zone's area the face covered**. That metric quietly lies. A small face in the dead centre of a wide top band covers maybe 7% of its area — comfortably "clear" — and the headline gets painted straight across it, which is exactly what the original fix was supposed to prevent.

The metric that works is **clearance**: the tallest contiguous strip of the zone the face does not cut through. Under it, that same top band scores 61 usable pixels against the ~220 the text needs, and correctly loses to a side column with 468.

**Lesson:** when a heuristic keeps producing the failure it was written to prevent, suspect the *metric*, not the thresholds. "How much is covered" and "is there room" sound like the same question and are not.

### 7.3 A slow/unstable network stalled the ffmpeg install

Not a code bug, but worth recording: the initial `winget install` of ffmpeg stalled at 0 bytes for several minutes. Switching to a direct `curl` download of the smaller "essentials" build (instead of the 253MB "full" build winget was fetching) finished in a couple of minutes. If you hit this: check whether the installer is actually making progress before assuming it's just slow.

### 7.4 Free-tier model quotas, and why the provider changed

The project originally ran its judgment stage on Google Gemini's free tier, and spent a lot of time fighting it:

- `gemini-2.5-flash` returned `404 NOT_FOUND: no longer available to new users` — while still appearing in `client.models.list()`, because it existed for *some* accounts.
- Back-to-back requests returned transient `503 UNAVAILABLE: high demand`, which needed a retry-with-backoff loop.
- Then a hard wall: `429 RESOURCE_EXHAUSTED ... limit: 20 ... quotaId: 'GenerateRequestsPerDayPerModel-FreeTier'` — 20 requests **per day**, which no amount of backoff fixes. The response even carried a `retryDelay` of a few seconds, which is actively misleading for a per-day quota.

Each of those got worked around in turn (aliases over pinned versions, backoff on 5xx only, switching to a model tier with a separate quota bucket). The workarounds held, but the pattern was the real signal: development kept stalling on quota rather than on the actual problem.

Consolidating both model stages onto kie.ai replaced all of it — one key, one balance, and pay-per-use instead of a daily cliff. The lessons survive the migration and are baked into `kie.py`: **retry 5xx, never 4xx** (a bad key doesn't become good on attempt three), and **read what an error actually says** before deciding whether waiting will help.

### 7.5 Duplicate frames made the shortlist meaningless

The shortlist was "sort by score, take the top 8". On a talking-head video that returns eight frames from the same few seconds of the same shot — the model was being handed a choice that wasn't a choice, and the "best frame" was effectively whichever near-identical still won a rounding contest.

The fix is `diversify()` (§4.3): a perceptual-hash and minimum-time-gap filter applied greedily, best-first, with a score-ordered backfill for videos too short or too static to satisfy it.

**Lesson:** "top N by score" assumes the candidates are meaningfully different. When they're sampled from a continuous source, that assumption is usually false, and the ranking hides it.

### 7.6 A 10-hour video silently ran for over an hour with no output

Someone pasted a **10-hour course video**. The web app sat on "Working..." indefinitely — no error, no progress. Digging in with `Get-CimInstance Win32_Process` turned up an ffmpeg child process running for over half an hour:
```
ffmpeg -ss 1080.15 -i source.mp4 -t 33844.7 -vf fps=0.00417 ... frame_%04d.jpg
```
`-t 33844.7` is a **9.4-hour** trim window. Nothing was broken — `MAX_SAMPLED_FRAMES` correctly kept the frame *count* at 150, exactly as designed. The problem was what the design never accounted for: **there was no limit on video length at all.** Sparse output sampling doesn't mean sparse decoding — most codecs still decode through the frames between the ones kept — so this was real wall-clock hours against a UI promising 30–90 seconds.

A compounding problem: Flask's dev server is **single-threaded by default**, so the long request froze the *entire* server, homepage included. From outside it looked like the app had crashed, not like one slow request.

Two fixes:

1. **Probe duration before downloading anything** — a metadata-only `extract_info(url, download=False)` for YouTube, `ffprobe` for uploads. Over `MAX_VIDEO_DURATION_SECONDS`, it raises immediately instead of starting a doomed job. Live streams, which report no fixed duration, are rejected the same way — there's no sensible "finished" point for an unbounded stream.
2. **`app.run(threaded=True)`**, so one slow `/generate` no longer blocks every other request.

**Lesson:** "cap the frame count" and "cap the video length" are different constraints, and capping one doesn't cap the other. Any stage whose cost scales with user-controlled input needs its own explicit ceiling, checked as early and as cheaply as possible — ideally before the expensive part starts.

---

## 8. What's Deliberately Not Here (Stretch Goals)

Left out on purpose to keep the project scoped and reliable, not because they'd be hard to bolt on:

- **A/B variants.** Generating 3 thumbnails per video and letting the user pick would be a real improvement and is a small change — `run_image_task` already returns a list of URLs. It's out because it triples the render cost per run.
- **Multiple faces.** Only the largest face is protected from overlap, so a two-person interview frame could still get a headline over the smaller face.
- **Avoiding *all* on-screen graphics, not just faces.** Placement avoids the face precisely, but knows nothing about a lower-third logo or a caption baked into the source video. A fuller fix would detect generally "busy" regions by edge density rather than just faces.
- **Transcript-aware frame *selection*.** Nearby speech currently informs the headline wording only; the director isn't told to prefer a frame because of a strong verbal claim at that instant. A fuller version would fold speech into frame scoring too.
- **A job queue for the web app.** `/generate` blocks for the full run. Fine for local single-user use; a real deployment would want background tasks and a progress stream — the `on_step` hook `pipeline.generate_thumbnail` already takes is the seam for it.
- **Caching renders.** Re-running the same video re-renders it. The work directory is keyed by video id, so short-circuiting on an existing output would be easy; it's out because during development you almost always *want* the re-run.
