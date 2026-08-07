# Thumbsmith

Give it a YouTube link or a video file, get back a thumbnail someone would actually click: the strongest frame, art-directed by a vision model, re-rendered by **GPT Image 2** on [kie.ai](https://kie.ai), with a headline placed where it will be read.

> Full deep-dive (every design decision, every bug we hit, why the code is shaped the way it is): **[HOW IT WORKS.md](HOW%20IT%20WORKS.md)**

## How it works, in 30 seconds

Three layers:

1. **Deterministic CV pre-filter (free, no API calls).** Get the video (yt-dlp at ≤720p, or a file you upload), sample ~100–150 frames with ffmpeg, and score each on sharpness, face size and position, exposure, colourfulness, and composition. De-duplicate near-identical frames so the shortlist spans the whole video, then keep the top 8.
2. **Art direction.** A vision model on kie.ai looks at those 8 frames and returns a *plan*, not just a pick: which frame, the headline, which side the subject should sit on, the accent colour, and the art direction for the render. One structured answer keeps every later stage in agreement.
3. **Render + compose.** GPT Image 2 re-renders the chosen frame image-to-image — graded, relit, decluttered, with clean negative space where the headline goes. The headline itself is drawn locally, because that is the only way to guarantee the exact words, at the right size, clear of the face and of YouTube's duration badge.

```
link or upload → sample → CV scoring → [top 8] → art director → GPT Image 2 → headline → thumbnail.jpg
```

It is image-**to**-image on purpose. Generating a fresh picture from a prompt invents a stranger in a scene that never happens; viewers bounce when the thumbnail does not match the first ten seconds.

**Optional add-on:** if `OPENAI_API_KEY` is set, `transcribe.py` transcribes the audio (Whisper) so the headline can quote what is actually said, not just what is visible. Skipped entirely if the key isn't set — no cost, no behaviour change.

## Run it

One key covers both model stages. Get it at [kie.ai/api-key](https://kie.ai/api-key).

**CLI:**
```powershell
$env:KIE_API_KEY = "your-key"

python main.py "https://youtu.be/..." -o thumbnail.jpg
python main.py --file "C:\clips\my-video.mp4" -o thumbnail.jpg
python main.py "https://youtu.be/..." --shortlist-only   # free stage only, no API calls
```

**Web app:**
```powershell
$env:KIE_API_KEY = "your-key"
python app.py   # → http://127.0.0.1:5000
```

The web UI has two tabs: paste a link, or drag in a video file (up to 512 MB; `.mp4 .mov .mkv .webm .avi .m4v`). Uploaded files are processed locally — only the 8 shortlisted frames ever leave your machine.

Requires `ffmpeg` and `ffprobe` on PATH; `pip install -r requirements.txt` handles the Python side. Videos over 2 hours are rejected on purpose (see [HOW IT WORKS.md §7.6](<HOW IT WORKS.md#76-a-10-hour-video-silently-ran-for-over-an-hour-with-no-output>)).

Copy [.env.example](.env.example) for the full list of settings.

## Project layout

```
thumbnail_picker/
  config.py       every tunable, in one place
  download.py     YouTube download + local file ingest → one VideoSource
  extract.py      frame sampling, CV scoring, de-duplication
  transcribe.py   optional Whisper pass
  kie.py          the only module that talks to kie.ai
  director.py     vision model → ThumbnailPlan
  generate.py     GPT Image 2 render
  compose.py      safe zones, placement, typography
  pipeline.py     the whole run, start to finish
main.py           CLI entrypoint
app.py            Flask web app
templates/, static/  web UI + generated output
```

`main.py` and `app.py` are both thin wrappers around `pipeline.generate_thumbnail` — they differ only in how they get a video and how they report progress, never in what the pipeline does.

## Why a model and not just a scoring script?

Heuristics can tell you a frame is *technically fine* — sharp, well exposed, has a face in it. They have no idea that a frame with "$3,459 saved" on screen is more clickable than an equally sharp one without it, and no opinion at all about what to write on it. That judgment is the model's job. Everything else is plain, free, deterministic code. See [HOW IT WORKS.md §2](<HOW IT WORKS.md#2-why-two-layers-the-core-design-decision>).

## Notes on placement

The things that quietly ruin a thumbnail, and what the compositor does about them:

- **The duration badge.** YouTube stamps the runtime over the bottom-right corner. Type is kept out of it.
- **The face.** Re-detected on the *final* image, not the source frame — the render is free to move the subject, so placement has to answer to where the face actually ended up.
- **Clearance, not overlap.** A zone is judged by the tallest face-free strip inside it. A small, centred face makes a wide top band useless even though it covers almost none of its area.
- **The outline.** Stroke width is counted when fitting text, so the headline never overhangs the safe margin YouTube crops on some surfaces.
