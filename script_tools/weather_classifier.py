"""Weather classification service for live CCTV frames.

Classifies daytime CCTV video frames as 'rain' or 'sunny' using the trained LightGBM
model from weather/kordon/models_ml/lgbm_weather.joblib. Low-light frames (e.g. night)
are intercepted early and categorized as 'unknown' without invoking the model.
"""

from pathlib import Path
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from script_tools.weather_ml import DATASETS, _crop
from script_tools.weather_tabular import TAB_FEATURES, extract_tabular_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "weather" / "kordon" / "models_ml" / "lgbm_weather.joblib"
MODEL_NAME = "LightGBM (Kordon)"
KORDON = DATASETS["kordon"]

# Minimum mean grayscale value for cropped ROI to be considered daytime.
# Kordon training frames have mean gray 90.1 - 117.8; 60.0 catches night/underlit frames safely.
LOW_LIGHT_GRAY = 60.0
MAX_FRAME_BYTES = 3_000_000

_MODEL: Any = None
_MODEL_LOCK = threading.Lock()
_MODEL_ERROR: Optional[str] = None


def _load_model() -> Any:
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        return None

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _MODEL_ERROR is not None:
            return None

        if not MODEL_PATH.exists():
            _MODEL_ERROR = f"Model file not found at {MODEL_PATH}"
            return None

        try:
            import joblib

            _MODEL = joblib.load(MODEL_PATH)
            return _MODEL
        except Exception as e:
            _MODEL_ERROR = f"{type(e).__name__}: {e}"
            return None


def classify_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Classify a single BGR OpenCV frame.

    Returns dict with keys:
      - status: 'success' | 'unavailable'
      - label: 'rain' | 'sunny' | 'unknown'
      - p_rain: float | None
      - confidence: float | None
      - brightness: float
      - model: str
      - reason?: str
    """
    if frame is None or frame.size == 0:
        return {"status": "error", "reason": "Frame tidak valid / kosong"}

    crop = _crop(frame, KORDON.crop)
    if crop is None or crop.size == 0:
        return {"status": "error", "reason": "Hasil crop frame kosong"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    if brightness < LOW_LIGHT_GRAY:
        return {
            "status": "success",
            "label": "unknown",
            "reason": "Frame terlalu gelap (malam) — di luar domain model",
            "p_rain": None,
            "confidence": None,
            "brightness": round(brightness, 1),
            "model": MODEL_NAME,
        }

    model = _load_model()
    if model is None:
        return {
            "status": "unavailable",
            "reason": f"Model cuaca tidak tersedia ({_MODEL_ERROR})",
            "brightness": round(brightness, 1),
            "model": MODEL_NAME,
        }

    feats = extract_tabular_features(crop, ds=KORDON)
    x = np.array([[feats[k] for k in TAB_FEATURES]], dtype=np.float32)

    # classes_ = [0, 1] where 0 = rain, 1 = sunny
    p_rain = float(model.predict_proba(x)[0][0])
    label = "rain" if p_rain >= 0.5 else "sunny"
    confidence = p_rain if label == "rain" else (1.0 - p_rain)

    return {
        "status": "success",
        "label": label,
        "p_rain": round(p_rain, 4),
        "confidence": round(confidence, 4),
        "brightness": round(brightness, 1),
        "model": MODEL_NAME,
    }


def classify_frame_bytes(data: bytes) -> Dict[str, Any]:
    """Classify raw image bytes (e.g. JPEG payload from POST request)."""
    if not data:
        return {"status": "error", "reason": "Frame kosong"}
    if len(data) > MAX_FRAME_BYTES:
        return {"status": "error", "reason": "Frame terlalu besar (>3MB)"}

    buf = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "error", "reason": "Frame tidak dapat didekode"}

    return classify_frame(frame)
