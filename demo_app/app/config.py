import os
import platform
import gc
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file if present

# Workaround for OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Disable Hugging Face Xet downloads
os.environ["HF_HUB_DISABLE_XET"] = "1"

if platform.system() == "Windows":
    base = r"D:\D\IMPORTANT\Dysarthria\ASR\huggingface_cache"
    # ── MUST be set before any huggingface/transformers imports ──────────────────
    os.environ["HF_HOME"] = base
    os.environ["TRANSFORMERS_CACHE"] = f"{base}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
    os.environ["TORCH_HOME"] = f"{base}/torch"

# Every value below is the one used for the paper (evaluation notebook, MODE="fixed").

# ── Model identifiers ──────────────────────────────────────────────────────────
FINE_TUNED_MODEL_ID = "ruch9265/distil-whisper-torgo"  # Whisper-tiny, encoder fine-tuned on TORGO
BASELINE_MODEL_ID = "openai/whisper-tiny"
WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base-960h"

SAMPLING_RATE = 16000
FEATURE_SIZE = 80

# ── ASR (Sect. 3.3, ASR Transcription) ─────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH_S", 30))

# ── CTC alignment ──────────────────────────────────────────────────────────────
# Each utterance is aligned alone: zero-padding a batch changes the outputs of
# wav2vec2-base (it was trained without an attention mask), and so the gaps.
CTC_PADDED_BATCH = False

# ── Pause formatting (Sect. 3.3, Pause Formatting) ─────────────────────────────
PAUSE_THRESHOLD = 0.30  # seconds; τ selected by the lift rule, validated by LOSO
LEADING_PAUSE_MARKER = False  # no marker for silence before the first word

# ── LLM post-correction (Sect. 3.3, LLM post correction) ───────────────────────
LLM_MODEL_ID = "openai/gpt-oss-120b"  # served by Groq
LLM_TEMPERATURE = 0.0
LLM_CONTEXT_WORDS = 4  # words kept on each side of a pause marker
LLM_MAX_ABS_EDITS = 2  # edit guard: reject if word edit distance > 2 ...
LLM_MAX_EDIT_RATIO = 0.40  # ... or > 40% of the region's words
LLM_RETRIES = 5
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")


# ── HF auth ─────────────────────────────────────────────────────────────────────
def hf_login() -> None:
    """Log into the Hugging Face Hub using HF_TOKEN from the environment, if present."""
    from huggingface_hub import login  # local import: keep heavy deps out of module load
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("logging into hugging face")
        login(token=hf_token)
        logger.info("logged into hugging face")
    else:
        logger.warning("HF_TOKEN not set — skipping Hugging Face login.")


def get_device_and_dtype():
    """
    Select the best available compute device.

    Returns:
        device: "mps", 0 (first CUDA GPU), or -1 (CPU) — matches the
                 `device` argument expected by transformers.pipeline().
        torch_dtype: float16 on CUDA (as in the paper's run on a T4),
                     float32 on MPS and CPU.
    """
    import torch  # local import: avoid importing torch just to read config constants

    gc.collect()

    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # MPS handles float16 in newer versions, but float32 is safest
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = 0  # hugging face pipeline expects an integer for CUDA device
        torch_dtype = torch.float16
        logger.info("Using CUDA GPU")
    else:
        device = -1  # hugging face pipeline expects an integer for CPU device
        torch_dtype = torch.float32
        logger.info("Using CPU")

    return device, torch_dtype


def torch_device_name(device) -> str:
    """Convert the pipeline-style device (0 / -1 / "mps") to a torch device string."""
    if device == "mps":
        return "mps"
    return "cpu" if device == -1 else f"cuda:{device}"
