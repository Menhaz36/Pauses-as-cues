import os
import platform
import gc
import logging
from dotenv import load_dotenv
from pydantic import SecretStr
from safetensors import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

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

# ── Model identifiers ──────────────────────────────────────────────────────────
FINE_TUNED_MODEL_ID = "ruch9265/distil-whisper-torgo"
BASELINE_MODEL_ID = "openai/whisper-tiny"
WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base-960h"

SAMPLING_RATE = 16000
FEATURE_SIZE = 80

# ── Default pipeline params ────────────────────────────────────────────────────
DEFAULT_PAUSE_THRESHOLD = 0.8  # seconds; τ used for pause-cue detection
# LLM_MODEL_ID = "llama-3.3-70b-versatile"
# groq_api_key = SecretStr(os.getenv("groq_api_key") or "")
LLM_MODEL_ID = "@cf/ibm-granite/granite-4.0-h-micro"
LLM_TEMPERATURE = 0.0
# openai_api_key = os.getenv("openai_api_key")
cf_api_token = os.getenv("CF_API_TOKEN")
cf_account_id = os.getenv("CF_ACCOUNT_ID")


#----------------------batch-------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
CHUNK_LENGTH_S = int(os.getenv("CHUNK_LENGTH_S", 30))


# ── HF auth ─────────────────────────────────────────────────────────────────────
def hf_login() -> None:
    """Log into the Hugging Face Hub using HF_TOKEN from the environment, if present."""
    from huggingface_hub import login  # local import: keep heavy deps out of module load
    load_dotenv()  # Load environment variables from .env file if present
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logging.info("logging into hugging face")
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
        torch_dtype: torch.float32 in all cases (fp16 is unstable for
                     whisper-tiny on MPS and unnecessary on this model size).
    """
    import torch  # local import: avoid importing torch just to read config constants

    gc.collect()

    # if torch.backends.mps.is_available():
    #     device = "mps"
    #     torch_dtype = torch.float32
    #     logger.info("Using MPS (Apple Silicon GPU)")
    # elif torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     device = 0
    #     torch_dtype = torch.float32
    #     logger.info("Using CUDA GPU")
    # else:
    #     device = -1
    #     torch_dtype = torch.float32
    #     logger.info("Using CPU")
    # return device, torch_dtype
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # MPS handles float16 in newer versions, but float32 is safest
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = 0  # hugging face pipeline expects an integer for CUDA device
        torch_dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32  # Optional GPU optimization
        logger.info("Using CUDA GPU")
    else:
        device = -1  # hugging face pipeline expects an integer for CPU device
        torch_dtype = torch.float32
        logger.info("Using CPU")

    return device, torch_dtype