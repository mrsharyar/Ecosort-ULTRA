"""
EcoSort Ultra Desktop - Core Engine
Extracted from EcoSortULTRAv4.py for desktop application use.
This module contains the AI inference engine without UI dependencies.
"""

# ═════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════════════
import gc
import hashlib
import time
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
MODEL_ID   = "microsoft/Florence-2-base"
MAX_DIM    = 640
OD_TOKENS  = 512
CAP_TOKENS = 256
NUM_BEAMS  = 1
MAX_CACHE_SIZE = 50
CLEANUP_INTERVAL = 5

# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class DetectionResult:
    """Represents a single detected object with classification."""
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    bin_type: str
    status: str
    tip: str

@dataclass
class ImageAnalysisResult:
    """Complete analysis result for a single image."""
    image: Image.Image
    annotated_image: Image.Image
    detections: List[DetectionResult]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

# ═════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED GLOBALS FOR SPEED
# ═════════════════════════════════════════════════════════════════════════════
GLOBAL_FONT: Optional[ImageFont.FreeTypeFont] = None

def _load_global_font():
    """Load font once globally to avoid per-image overhead."""
    global GLOBAL_FONT
    if GLOBAL_FONT is not None:
        return GLOBAL_FONT
    try:
        GLOBAL_FONT = ImageFont.truetype("arial.ttf", size=18)
    except Exception:
        try:
            GLOBAL_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=18)
        except Exception:
            GLOBAL_FONT = ImageFont.load_default()
    return GLOBAL_FONT

_SORTED_KEYS: List[str] = []
_RULES_LOOKUP: Dict[str, Dict[str, str]] = {}

# ═════════════════════════════════════════════════════════════════════════════
# DEVICE & MODEL SETUP
# ═════════════════════════════════════════════════════════════════════════════
DEVICE: torch.device
DTYPE: torch.dtype
FLORENCE_MODEL = None
FLORENCE_PROCESSOR = None
ENGINE_STATUS = "LOADING…"

def _detect_device() -> Tuple[torch.device, torch.dtype]:
    """Detect best available compute device."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda"), torch.float16
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), torch.bfloat16
    except Exception:
        pass
    return torch.device("cpu"), torch.float32

DEVICE, DTYPE = _detect_device()

def _load_model():
    """Load Florence-2 model with retry logic."""
    global FLORENCE_MODEL, FLORENCE_PROCESSOR, ENGINE_STATUS
    for attn in ("eager", None):
        try:
            kwargs: Dict = dict(torch_dtype=DTYPE, trust_remote_code=True)
            if attn:
                kwargs["attn_implementation"] = attn
            FLORENCE_MODEL = (
                Florence2ForConditionalGeneration
                .from_pretrained(MODEL_ID, **kwargs)
                .to(DEVICE)
                .eval()
            )
            FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )
            ENGINE_STATUS = (
                f"{'Florence-2-Large' if 'large' in MODEL_ID else 'Florence-2-Base'} ✓ "
                f"| {str(DEVICE).upper()} | {DTYPE}"
            )
            return
        except Exception as exc:
            print(f"[WARN] Model load attempt failed ({attn=}): {exc}")
    ENGINE_STATUS = "LOAD FAILED — check GPU memory and network"

_load_model()
print(f"[EcoSort] {ENGINE_STATUS}")
