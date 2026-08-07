import os
from dataclasses import dataclass

import yt_dlp

from . import config


@dataclass
class DownloadResult:
    video_path: str
    video_id: str
    title: str
    duration: float


def _probe_duration(url: str) -> float:
    """Fetch just the video's metadata (no download) so we can reject overly long videos in
    seconds instead of after downloading gigabytes and grinding through hours of ffmpeg decoding."""
    opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return float(info.get("duration") or 0)


def download_video(url: str) -> DownloadResult:
    """Download a YouTube video capped at MAX_DOWNLOAD_HEIGHT and return its local path + metadata."""
    duration = _probe_duration(url)
    if duration <= 0:
        raise RuntimeError(
            "Could not determine this video's length (it may be a live stream). "
            "This tool needs a regular, finished video with a fixed duration."
        )
    if duration > config.MAX_VIDEO_DURATION_SECONDS:
        raise RuntimeError(
            f"This video is about {duration / 3600:.1f} hours long — longer than the "
            f"{config.MAX_VIDEO_DURATION_SECONDS / 3600:.0f}-hour limit this tool supports. "
            "Long videos take a very long time to download and process (hours, not seconds). "
            "Try a shorter video, or raise MAX_VIDEO_DURATION_SECONDS in config.py if you really "
            "want to process it (and are prepared to wait)."
        )

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

    return DownloadResult(
        video_path=video_path,
        video_id=info["id"],
        title=info.get("title", ""),
        duration=float(info.get("duration", 0)),
    )
