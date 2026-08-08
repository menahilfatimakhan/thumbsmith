import os

# Load a .env file if the user made one, so KIE_API_KEY can live in a file instead of
# being exported in every new shell. Optional: python-dotenv is not a hard dependency,
# and real environment variables always win over anything in .env.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL_SECONDS = 2
MAX_SAMPLED_FRAMES = 150
SKIP_INTRO_OUTRO_FRACTION = 0.03
MAX_DOWNLOAD_HEIGHT = 720

# Hard cap on video length. MAX_SAMPLED_FRAMES already bounds *frame count* regardless of
# duration, but a multi-hour video still means downloading gigabytes and ffmpeg decoding
# through the whole thing just to pull out those sparse frames — real wall-clock hours, not
# the ~30-90 seconds this tool is designed around. Checked before downloading anything.
MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60  # 2 hours

# Largest video file accepted through the web uploader, in bytes.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MB
ALLOWED_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

SHORTLIST_SIZE = 8

OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720

# ---------------------------------------------------------------------------
# Heuristic frame scoring
# ---------------------------------------------------------------------------
MIN_LAPLACIAN_VAR = 20

WEIGHT_SHARPNESS = 0.25
WEIGHT_FACE = 0.35
WEIGHT_EXPOSURE = 0.15
WEIGHT_COLORFULNESS = 0.15
WEIGHT_COMPOSITION = 0.10

# A face has to actually fill some of the frame to carry a thumbnail. A face box smaller
# than this fraction of the frame area reads as "person somewhere in a wide shot", which
# is dead weight at the 210x118px size a thumbnail is usually seen at.
IDEAL_FACE_AREA_FRACTION = 0.12
MIN_USEFUL_FACE_AREA_FRACTION = 0.01

# Two frames three seconds apart in a talking-head video are the same picture. Without
# de-duplication the shortlist collapses onto one moment and the director has no real
# choice to make. Frames closer than this in *appearance* are treated as duplicates.
DEDUPE_HASH_DISTANCE = 6
# ...and candidates are also forced apart in time, so the shortlist spans the whole video.
MIN_CANDIDATE_GAP_SECONDS = 8.0

# ---------------------------------------------------------------------------
# kie.ai — https://docs.kie.ai
# One API key covers both stages below. Set KIE_API_KEY in the environment.
# ---------------------------------------------------------------------------
KIE_API_KEY_ENV = "KIE_API_KEY"

# The file-upload service lives on a different host to the rest of the API. docs.kie.ai
# documents it under api.kie.ai, which 404s — this is the host that actually serves it.
KIE_UPLOAD_URL = "https://kieai.redpandaai.co/api/file-base64-upload"
KIE_CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_RECORD_INFO_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"
KIE_CHAT_URL_TEMPLATE = "https://api.kie.ai/{model}/v1/chat/completions"

# Stage 2a — the "art director": a vision model looks at the shortlist and returns a
# structured plan (which frame, what headline, where the text goes, what to fix).
KIE_CHAT_MODEL = os.environ.get("KIE_CHAT_MODEL", "gpt-5-2")
KIE_CHAT_REASONING_EFFORT = "low"

# Stage 2b — the image model that actually renders the thumbnail. GPT Image 2, driven
# image-to-image off the chosen frame so the real subject survives into the output
# instead of being replaced by an invented stock person.
KIE_IMAGE_MODEL = os.environ.get("KIE_IMAGE_MODEL", "gpt-image-2-image-to-image")
KIE_IMAGE_ASPECT_RATIO = "16:9"

# GPT Image 2 has no quality parameter — resolution is the only quality/cost selector.
# On kie.ai that is 6 credits ($0.03) at 1K, 10 ($0.05) at 2K, 16 ($0.08) at 4K.
#
# 1K is the right default here, not just the cheapest: a 16:9 1K render still comes back
# larger than the 1280x720 we export, so the extra pixels of a 2K render are thrown away
# by the downscale in compose.py. Raise it only if you also raise OUTPUT_WIDTH/HEIGHT.
KIE_IMAGE_RESOLUTION = os.environ.get("KIE_IMAGE_RESOLUTION", "1K")

# ---------------------------------------------------------------------------
# YouTube access
# ---------------------------------------------------------------------------
# YouTube bot-gates datacenter IPs. The exact same video and the exact same yt-dlp version
# that work from a home connection get "Sign in to confirm you're not a bot" from a cloud
# host, and no player_client trick gets around it — the IP is what is flagged.
#
# Two ways out, both optional:
#   YTDLP_COOKIES_FILE  path to a Netscape-format cookies.txt exported from a logged-in
#                       browser. Effective, but cookies expire after days or weeks, and
#                       Google may flag an account whose cookies appear from a server in
#                       another country. Use a throwaway account, never a primary one.
#   YTDLP_PROXY         a residential or rotating proxy URL. Costs money, but it is the
#                       only fix that keeps working unattended.
#
# With neither set, the link tab simply fails on a blocked host and the upload tab, which
# never touches YouTube, keeps working.
YTDLP_COOKIES_FILE = os.environ.get("YTDLP_COOKIES_FILE", "")
YTDLP_PROXY = os.environ.get("YTDLP_PROXY", "")

KIE_UPLOAD_PATH = "images/thumbsmith"
KIE_REQUEST_TIMEOUT_SECONDS = 120
KIE_POLL_INTERVAL_SECONDS = 3
KIE_POLL_TIMEOUT_SECONDS = 420
KIE_MAX_RETRIES = 3
KIE_RETRY_BACKOFF_SECONDS = 3

# Downloading the finished render gets more attempts than anything else: the image is
# already rendered and already paid for, so giving up here means paying full price for the
# worst possible output (the ungraded source frame).
KIE_DOWNLOAD_RETRIES = 5

# Frames are downscaled before upload — the director only needs enough pixels to judge
# expression and composition, and 8 full-size stills is a lot of wasted bandwidth.
DIRECTOR_IMAGE_MAX_WIDTH = 800
# The chosen frame goes to the image model at full width, since that one *is* the input
# the render is built from.
RENDER_IMAGE_MAX_WIDTH = 1280

# Let GPT Image 2 paint the headline itself instead of compositing it locally. Off by
# default: locally drawn text is the only way to guarantee the exact words, exact
# placement, and a clean 1280x720 export every single run.
LET_MODEL_RENDER_TEXT = False

# If the render stage fails (quota, outage, a task that comes back empty), fall back to
# the colour-graded original frame rather than failing the whole run.
FALLBACK_TO_RAW_FRAME = True

# ---------------------------------------------------------------------------
# Typography / layout
# ---------------------------------------------------------------------------
# Anton is the typeface this is designed around: heavy, condensed, geometric, and the de
# facto look of high-CTR thumbnails. Arial Black and Impact (the old defaults, and what
# every system font list falls back to) are what make a thumbnail read as amateur —
# Impact in particular now signals "2010 meme" rather than "designed".
#
# Committed to the repo (167 KB) so a deploy needs no network fetch to render type
# correctly, and downloaded on first use if it is ever missing.
FONT_PATH = "fonts/Anton-Regular.ttf"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"

# Only reached if Anton cannot be downloaded. Override the whole thing with
# THUMBNAIL_FONT_PATH to use a brand typeface instead.
FONT_CANDIDATES = [
    os.environ.get("THUMBNAIL_FONT_PATH", ""),
    FONT_PATH,
    # Windows
    r"C:\Windows\Fonts\ariblk.ttf",      # Arial Black
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]

# Reference thumbnails set their lines almost touching. 1.06 leaves the airy, default-ish
# gap that reads as "text placed by software" rather than "type set by a designer".
# Applied to the font's ascent+descent, not to its point size.
LINE_HEIGHT_RATIO = 0.86

# Paint a filled block behind the accent word instead of just recolouring it. This is the
# single strongest "a designer made this" cue in the reference set.
ACCENT_AS_BLOCK = True

# Outline thickness as a fraction of the font size.
#
# The old value was 1/11, a heavy black keyline. That single choice is most of what made
# output read as amateur: it is the Impact-meme look, and none of the modern reference
# thumbnails use it. The better ones carry no outline whatsoever and separate the type
# from the picture with negative space and a soft shadow instead. A hairline is kept so
# text stays legible if the render hands back a busier background than expected.
STROKE_RATIO = 1 / 26

# Drop shadow does the work the outline used to. Offset and blur are both relative to the
# font size so the treatment holds at any headline length.
SHADOW_OFFSET_RATIO = 1 / 22
SHADOW_BLUR_RATIO = 1 / 9
SHADOW_ALPHA = 165

# YouTube stamps the duration badge over the bottom-right corner of every thumbnail, and
# crops the edges on some surfaces. Keep type out of these fractions of the frame.
SAFE_MARGIN_FRACTION = 0.045
DURATION_BADGE_W_FRACTION = 0.22
DURATION_BADGE_H_FRACTION = 0.17

DEFAULT_ACCENT_COLOR = "#FFD400"

WORK_DIR = "work"

# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
# OpenCV 5.0 removed the old Haar cascade CascadeClassifier + bundled XML files.
# Face detection now uses the DNN-based YuNet model instead.
FACE_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# ---------------------------------------------------------------------------
# Optional transcription
# ---------------------------------------------------------------------------
# Only used if OPENAI_API_KEY is set. Adds "what's being said at this moment" context to
# each candidate frame so headlines can be grounded in actual speech, not just what's
# visible. Costs real money (~$0.006/minute of audio) — there's no free tier for this.
TRANSCRIBE_MODEL = "whisper-1"
TRANSCRIBE_WINDOW_SECONDS = 6.0
