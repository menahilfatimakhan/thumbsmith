SAMPLE_INTERVAL_SECONDS = 2
MAX_SAMPLED_FRAMES = 150
SKIP_INTRO_OUTRO_FRACTION = 0.03
MAX_DOWNLOAD_HEIGHT = 720

# Hard cap on video length. MAX_SAMPLED_FRAMES already bounds *frame count* regardless of
# duration, but a multi-hour video still means downloading gigabytes and ffmpeg decoding
# through the whole thing just to pull out those sparse frames — real wall-clock hours, not
# the ~30-90 seconds this tool is designed around. Checked before downloading anything.
MAX_VIDEO_DURATION_SECONDS = 2 * 60 * 60  # 2 hours

SHORTLIST_SIZE = 8

OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720

MIN_LAPLACIAN_VAR = 20
WEIGHT_SHARPNESS = 0.4
WEIGHT_FACE = 0.4
WEIGHT_EXPOSURE = 0.2

# "gemini-flash-latest" has a much smaller free-tier daily quota (20 requests/day) and
# exhausts fast during development. "gemini-flash-lite-latest" is a separate, higher-quota
# tier that's still plenty capable for picking one of 8 images + writing a short caption.
GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_IMAGE_MAX_WIDTH = 800

WORK_DIR = "work"

# OpenCV 5.0 removed the old Haar cascade CascadeClassifier + bundled XML files.
# Face detection now uses the DNN-based YuNet model instead.
FACE_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Optional: only used if OPENAI_API_KEY is set. Adds "what's being said at this moment"
# context to each candidate frame so captions can be grounded in actual speech, not just
# what's visible. Costs real money (~$0.006/minute of audio) — there's no free tier for this.
TRANSCRIBE_MODEL = "whisper-1"
TRANSCRIBE_WINDOW_SECONDS = 6.0
