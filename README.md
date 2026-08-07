# YouTube Thumbnail Picker

Paste a YouTube link, get back a designed thumbnail: the best frame, picked by AI, cropped to 1280×720, color-punched, with a bold caption overlaid.

> Full deep-dive (every design decision, every bug we hit, why the code is shaped the way it is): **[HOW IT WORKS.md](HOW%20IT%20WORKS.md)**

## How it works, in 30 seconds

Two layers, not one:

1. **Deterministic CV pre-filter (free, no API calls).** Download the video (yt-dlp, capped at 720p), sample ~100–150 frames (ffmpeg), score each one on sharpness/face-detection/exposure, keep the top 8.
2. **AI judgment (the one place a model is actually needed).** Google Gemini looks at those 8 frames, picks the one a human would actually click, and writes a punchy caption — the thing plain computer vision can't do, because it has no idea what's *interesting*, only what's *sharp*.

The result is then cropped/resized/color-boosted, and the caption is placed to avoid the detected face, on a translucent panel for contrast.

```
YouTube URL → download → sample frames → CV scoring → [top 8] → Gemini picks + captions → compose → thumbnail.jpg
```

**Optional add-on:** if `OPENAI_API_KEY` is set, `transcribe.py` transcribes the audio (OpenAI Whisper) so captions can be grounded in what's actually being said, not just what's visible. Skipped entirely if the key isn't set — no cost, no behavior change.

## Run it

**CLI:**
```powershell
$env:GEMINI_API_KEY = "your-key"
python main.py "https://youtu.be/..." -o thumbnail.jpg
python main.py "https://youtu.be/..." --shortlist-only   # test the free part only, no API call
```

**Web app:**
```powershell
$env:GEMINI_API_KEY = "your-key"
python app.py   # → http://127.0.0.1:5000
```

Requires `ffmpeg` on PATH (`pip install -r requirements.txt` handles the Python side). Gemini's free tier needs no billing setup. Videos over 2 hours are rejected on purpose (see [HOW IT WORKS.md §7.6](<HOW IT WORKS.md#76-a-10-hour-video-silently-ran-for-over-an-hour-with-no-output>)).

## Project layout

```
thumbnail_picker/   config.py, download.py, extract.py, transcribe.py, agent.py, compose.py
main.py             CLI entrypoint
app.py              Flask web app
templates/, static/ Web UI + generated thumbnail output
```

## Why an "agent" and not just a scoring script?

Because heuristics (blur detection, face detection, exposure) can tell you a frame is *technically fine*, but they have no idea a frame with "$3,459 saved" written on screen is more clickable than an equally sharp, equally well-lit frame without it. That judgment call is Gemini's job; everything else is plain, free, deterministic code. See [HOW IT WORKS.md §2](<HOW IT WORKS.md#2-why-two-layers-the-core-design-decision>) for the full reasoning.
