"""Getting a video onto disk — either pulled from YouTube, or handed to us directly.

Both paths end at the same ``VideoSource``, so nothing downstream needs to care which one
the user picked.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import yt_dlp

from . import config


@dataclass
class VideoSource:
    video_path: str
    video_id: str
    title: str
    duration: float
    origin: str  # "youtube" or "upload"


# Kept as an alias so existing callers and docs that say DownloadResult still work.
DownloadResult = VideoSource


def _too_long_error(duration: float) -> RuntimeError:
    return RuntimeError(
        f"This video is about {duration / 3600:.1f} hours long, over the "
        f"{config.MAX_VIDEO_DURATION_SECONDS / 3600:.0f}-hour limit this tool supports. "
        "Long videos take a very long time to download and process (hours, not seconds). "
        "Try a shorter video, or raise MAX_VIDEO_DURATION_SECONDS in config.py if you really "
        "want to process it (and are prepared to wait)."
    )


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def _probe_duration(url: str) -> float:
    """Fetch just the video's metadata (no download) so we can reject overly long videos in
    seconds instead of after downloading gigabytes and grinding through hours of ffmpeg decoding."""
    opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return float(info.get("duration") or 0)


def download_video(url: str) -> VideoSource:
    """Download a YouTube video capped at MAX_DOWNLOAD_HEIGHT and return its local path + metadata."""
    duration = _probe_duration(url)
    if duration <= 0:
        raise RuntimeError(
            "Could not determine this video's length (it may be a live stream). "
            "This tool needs a regular, finished video with a fixed duration."
        )
    if duration > config.MAX_VIDEO_DURATION_SECONDS:
        raise _too_long_error(duration)

    out_template = os.path.join(config.WORK_DIR, "%(id)s", "source.%(ext)s")
    ydl_opts = {
        "format": f"bestvideo[height<={config.MAX_DOWNLOAD_HEIGHT}]+bestaudio/best[height<={config.MAX_DOWNLOAD_HEIGHT}]",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        # merge_output_format can change the extension after download
        if not os.path.exists(video_path):
            base, _ = os.path.splitext(video_path)
            video_path = base + ".mp4"

    return VideoSource(
        video_path=video_path,
        video_id=info["id"],
        title=info.get("title", ""),
        duration=float(info.get("duration", 0)),
        origin="youtube",
    )


# ---------------------------------------------------------------------------
# Direct upload
# ---------------------------------------------------------------------------

def probe_local_duration(path: str) -> float:
    """Read a local file's duration with ffprobe (ffmpeg is already a hard dependency)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            check=True, capture_output=True, text=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _local_video_id(path: str, display_name: str) -> str:
    """A stable, filesystem-safe id for an uploaded file, so re-uploading it reuses its work dir."""
    stat = os.stat(path)
    digest = hashlib.sha1(f"{display_name}:{stat.st_size}".encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", os.path.splitext(display_name)[0]).strip("-").lower()[:24]
    return f"upload-{slug or 'video'}-{digest}"


def ingest_local_video(path: str, display_name: str | None = None, move: bool = False) -> VideoSource:
    """Adopt a video file the user supplied directly and return the same shape as a download.

    The file lands in the work directory so the pipeline writes its frames next to the video
    exactly as it does for a downloaded one. It is copied by default, leaving the user's own
    file untouched; the web uploader passes move=True because its source is a throwaway temp
    file and copying half a gigabyte twice is pure waste.
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"No such video file: {path}")

    display_name = display_name or os.path.basename(path)
    extension = os.path.splitext(display_name)[1].lower()
    if extension not in config.ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))
        raise RuntimeError(f"'{extension or 'that file type'}' is not supported. Upload one of: {allowed}")

    size = os.path.getsize(path)
    if size > config.MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"That file is {size / 1024 / 1024:.0f} MB, over the "
            f"{config.MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB limit. Trim it or export it smaller."
        )

    duration = probe_local_duration(path)
    if duration <= 0:
        raise RuntimeError(
            "Could not read that file as a video. It may be corrupt, still uploading, or "
            "not actually a video file."
        )
    if duration > config.MAX_VIDEO_DURATION_SECONDS:
        raise _too_long_error(duration)

    video_id = _local_video_id(path, display_name)
    video_dir = os.path.join(config.WORK_DIR, video_id)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"source{extension}")

    if os.path.abspath(path) != os.path.abspath(video_path):
        if move:
            shutil.move(path, video_path)
        else:
            shutil.copyfile(path, video_path)

    return VideoSource(
        video_path=video_path,
        video_id=video_id,
        title=os.path.splitext(display_name)[0],
        duration=duration,
        origin="upload",
    )
