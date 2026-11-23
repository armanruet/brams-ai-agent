"""Store verified archive PNGs; only opaque IDs go into model messages."""

from io import BytesIO
from pathlib import Path
import re
import shutil
import warnings
import hashlib
from uuid import uuid4


SPECTROGRAM_DIR = Path(__file__).resolve().parent.parent / ".artifacts" / "spectrograms"
WAV_DIR = Path(__file__).resolve().parent.parent / ".artifacts" / "wav"
DETECTION_DIR = Path(__file__).resolve().parent.parent / ".artifacts" / "detections"


def _verify_png(content: bytes) -> None:
    from PIL import Image

    if not content or len(content) > 20 * 1024 * 1024:
        raise ValueError("PNG image is empty or exceeds 20 MB.")
    with Image.open(BytesIO(content)) as image:
        if image.format != "PNG":
            raise ValueError("The image response is not a PNG image.")
        image.verify()


def save_spectrogram(content: bytes) -> str:
    _verify_png(content)
    SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid4().hex
    (SPECTROGRAM_DIR / f"{artifact_id}.png").write_bytes(content)
    return artifact_id


def spectrogram_path(artifact_id: str) -> Path | None:
    if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
        return None
    path = SPECTROGRAM_DIR / f"{artifact_id}.png"
    return path if path.is_file() else None


def save_detection_image(content: bytes) -> str:
    _verify_png(content)
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid4().hex
    (DETECTION_DIR / f"{artifact_id}.png").write_bytes(content)
    return artifact_id


def detection_path(artifact_id: str) -> Path | None:
    if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
        return None
    path = DETECTION_DIR / f"{artifact_id}.png"
    return path if path.is_file() else None


def save_wav_file(source: Path) -> dict:
    """Keep original bytes and create a separate PCM16 listening preview."""
    import numpy as np
    from scipy.io import wavfile

    if not source.is_file() or not 12 <= source.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("WAV is missing, empty or exceeds 64 MB.")
    with warnings.catch_warnings():
        # BRAMS BRA1/BRA2 scientific metadata chunks are preserved in the original.
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        rate, data = wavfile.read(source)
    if rate <= 0 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("WAV contains no valid audio samples.")
    preview = data.astype(np.float64)
    if data.dtype == np.uint8:
        preview -= 128
    peak = float(np.max(np.abs(preview)))
    if peak:
        preview /= peak
    preview = (preview * 32767).astype(np.int16)
    artifact_id = uuid4().hex
    directory = WAV_DIR / artifact_id
    directory.mkdir(parents=True)
    try:
        shutil.copyfile(source, directory / "original.wav")
        wavfile.write(directory / "preview.wav", rate, preview)
    except Exception:
        shutil.rmtree(directory)
        raise
    return {"artifact_id": artifact_id, "sample_rate_hz": int(rate),
            "sample_count": len(data),
            "size_bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "channels": 1 if data.ndim == 1 else int(data.shape[1]),
            "duration_seconds": len(data) / rate,
            "preview_note": "Player uses a peak-normalized PCM16 listening preview. Download the original WAV to retain the unchanged scientific data and metadata."}


def wav_path(artifact_id: str, *, preview: bool = False) -> Path | None:
    if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
        return None
    path = WAV_DIR / artifact_id / ("preview.wav" if preview else "original.wav")
    return path if path.is_file() else None
