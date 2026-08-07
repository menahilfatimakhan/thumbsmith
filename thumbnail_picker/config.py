import os

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

KIE_UPLOAD_URL = "https://api.kie.ai/api/file-base64-upload"
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
KIE_IMAGE_RESOLUTION = "2K"

KIE_UPLOAD_PATH = "images/thumbsmith"
KIE_REQUEST_TIMEOUT_SECONDS = 120
KIE_POLL_INTERVAL_SECONDS = 3
KIE_POLL_TIMEOUT_SECONDS = 420
KIE_MAX_RETRIES = 3
KIE_RETRY_BACKOFF_SECONDS = 3

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
# First font that exists on the box wins. Override with THUMBNAIL_FONT_PATH.
FONT_CANDIDATES = [
    os.environ.get("THUMBNAIL_FONT_PATH", ""),
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
