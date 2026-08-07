import argparse
import os
import sys

from thumbnail_picker import agent, compose, config, download, extract, transcribe


def run(url: str, output: str, shortlist_only: bool) -> None:
    print(f"Downloading video (capped at {config.MAX_DOWNLOAD_HEIGHT}p)...")
    result = download.download_video(url)
    print(f"  title={result.title!r} duration={result.duration:.0f}s -> {result.video_path}")

    video_dir = os.path.dirname(result.video_path)
    frames_dir = os.path.join(video_dir, "frames")

    print("Sampling and scoring frames...")
    shortlist = extract.get_shortlist(result.video_path, result.duration, frames_dir)
    if not shortlist:
        print("No usable frames passed the sharpness/exposure floor. Try a different video.")
        sys.exit(1)

    print(f"Shortlisted {len(shortlist)} candidates:")
    for c in shortlist:
        print(f"  {c.id}  t={c.timestamp:6.1f}s  score={c.score:.2f}  "
              f"sharp={c.sharpness:.2f}  face={c.has_face}  exposure={c.exposure_score:.2f}")

    if shortlist_only:
        print(f"\n--shortlist-only: inspect frames in {frames_dir}")
        return

    transcript_segments = []
    if transcribe.is_available():
        print("Transcribing audio with OpenAI Whisper for extra caption context...")
        transcript_segments = transcribe.transcribe(result.video_path)
        print(f"  got {len(transcript_segments)} transcript segments" if transcript_segments
              else "  transcription returned nothing — continuing on visuals alone")

    print("\nAsking Gemini to pick the best frame and write a caption...")
    selection = agent.pick_best_frame(shortlist, transcript_segments)
    chosen = next(c for c in shortlist if c.id == selection.candidate_id)
    print(f"  chosen: {chosen.id} (t={chosen.timestamp:.1f}s) — {selection.reason}")
    print(f"  caption: \"{selection.caption}\"")

    print(f"Composing final thumbnail -> {output}")
    compose.make_thumbnail(chosen.path, output, selection.caption, chosen.face_box)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick the best frame from a YouTube video and make a thumbnail.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", default="thumbnail.jpg", help="Output thumbnail path")
    parser.add_argument("--shortlist-only", action="store_true",
                         help="Stop after Stage 1 (download+sample+shortlist) — no Gemini API call")
    args = parser.parse_args()

    run(args.url, args.output, args.shortlist_only)


if __name__ == "__main__":
    main()
