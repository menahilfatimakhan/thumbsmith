import argparse
import os
import sys

from thumbnail_picker import config, download, extract, kie, pipeline


def _load_source(url: str | None, file_path: str | None) -> download.VideoSource:
    if file_path:
        print(f"Reading local video {file_path}...")
        return download.ingest_local_video(file_path)

    print(f"Downloading video (capped at {config.MAX_DOWNLOAD_HEIGHT}p)...")
    return download.download_video(url)


def _print_shortlist(shortlist: list[extract.Candidate]) -> None:
    print(f"Shortlisted {len(shortlist)} candidates:")
    for c in shortlist:
        print(f"  {c.id}  t={c.timestamp:6.1f}s  score={c.score:.2f}  sharp={c.sharpness:.2f}  "
              f"face={c.has_face} ({c.face_fraction:.1%})  colour={c.colorfulness:.2f}  "
              f"comp={c.composition:.2f}")


def run(url: str | None, file_path: str | None, output: str, shortlist_only: bool) -> None:
    source = _load_source(url, file_path)
    print(f"  title={source.title!r} duration={source.duration:.0f}s -> {source.video_path}")

    if shortlist_only:
        frames_dir = os.path.join(os.path.dirname(source.video_path), "frames")
        print("Sampling and scoring frames...")
        shortlist = extract.get_shortlist(source.video_path, source.duration, frames_dir)
        if not shortlist:
            print("No usable frames passed the sharpness/exposure floor. Try a different video.")
            sys.exit(1)
        _print_shortlist(shortlist)
        print(f"\n--shortlist-only: inspect frames in {frames_dir}")
        return

    result = pipeline.generate_thumbnail(source, output, on_step=print)

    _print_shortlist(result.shortlist)
    print(f"\nChosen: {result.chosen.id} (t={result.chosen.timestamp:.1f}s) — {result.plan.reason}")
    print(f'  headline: "{result.plan.headline}"  accent={result.plan.accent_word or "none"} '
          f'{result.plan.accent_color}')
    print(f"  subject on the {result.plan.subject_side}, headline on the {result.text_side}")
    print(f"  image: {'rendered by ' + config.KIE_IMAGE_MODEL if result.rendered else 'original frame'}")
    for warning in result.warnings:
        print(f"  note: {warning}")
    print(f"Done -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a YouTube video or a local video file into a finished thumbnail.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("url", nargs="?", help="YouTube video URL")
    source_group.add_argument("-f", "--file", help="Path to a local video file instead of a URL")
    parser.add_argument("-o", "--output", default="thumbnail.jpg", help="Output thumbnail path")
    parser.add_argument("--shortlist-only", action="store_true",
                        help="Stop after Stage 1 (get video + sample + shortlist) — no kie.ai calls")
    args = parser.parse_args()

    try:
        run(args.url, args.file, args.output, args.shortlist_only)
    except (kie.KieError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
