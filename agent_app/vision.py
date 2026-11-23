"""Local YOLOv8/YOLOv11 meteor echo detection for BRAMS radio spectrograms.

Follows Fernandes et al. (2026): Raw RGB spectrogram inputs (no threshold
denoising), resized/letterboxed to 640x640 normalized to [0, 1], executed
locally via ONNX Runtime (primary) or Ultralytics (fallback). Image bytes and
filesystem paths remain strictly outside LLM messages.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from io import BytesIO
import json
import math
from pathlib import Path
import threading
from time import perf_counter
from typing import Any

from .artifacts import save_detection_image
from .config import DEFAULT_VISION_MODEL, PROJECT_ROOT


VISION_LOCK = threading.Lock()
_CACHED_DETECTOR: tuple[str, int, Any] | None = None

# Radio Meteor Zoo canonical spectrogram parameters (Calders et al., 2016/2026; Fernandes et al., 2026)
RMZ_WIDTH = 864
RMZ_HEIGHT = 595
RMZ_FFT_SIZE = 16384
RMZ_OVERLAP_RATIO = 0.90
RMZ_BAND_HZ = 200.0


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_vision_model_path(model_path: str | Path | None = None) -> Path:
    raw = Path(model_path or DEFAULT_VISION_MODEL)
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw)


def load_model_manifest(model_path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "model_name": model_path.stem,
        "architecture": "YOLOv11-n",
        "input_mode": "archive_png",
        "imgsz": 640,
        "resize_mode": "letterbox",
        "class_names": {"0": "meteor"},
        "preprocessing": "raw_rgb_0_1_no_threshold_denoising",
    }
    candidates = [
        model_path.with_suffix(".manifest.json"),
        model_path.parent / "model_manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                loaded = json.loads(candidate.read_text())
                if isinstance(loaded, dict):
                    return {**defaults, **loaded}
            except (OSError, ValueError):
                continue
    return defaults


def vision_setup_status(model_path: str | Path | None = None) -> dict[str, Any]:
    resolved = resolve_vision_model_path(model_path)
    manifest = load_model_manifest(resolved)
    suffix = resolved.suffix.lower()

    if find_spec("numpy") is None or find_spec("PIL") is None:
        return {
            "ready": False,
            "code": "dependency_missing",
            "message": "Install Pillow and NumPy with: python -m pip install -r requirements-vision.txt",
            "model_path": resolved.name,
        }

    has_ort = find_spec("onnxruntime") is not None
    has_ultralytics = find_spec("ultralytics") is not None

    if suffix == ".onnx" and not (has_ort or has_ultralytics):
        return {
            "ready": False,
            "code": "dependency_missing",
            "message": "Install ONNX Runtime with: python -m pip install -r requirements-vision.txt",
            "model_path": resolved.name,
        }
    if suffix == ".pt" and not has_ultralytics:
        return {
            "ready": False,
            "code": "dependency_missing",
            "message": "PyTorch .pt checkpoints require ultralytics (or export to .onnx foronnxruntime).",
            "model_path": resolved.name,
        }
    if suffix not in {".onnx", ".pt"}:
        return {
            "ready": False,
            "code": "unsupported_model_format",
            "message": f"Unsupported vision model extension '{suffix}'. Use .onnx or .pt.",
            "model_path": resolved.name,
        }
    if not resolved.is_file():
        return {
            "ready": False,
            "code": "weights_missing",
            "message": (
                f"Vision detector weights not found ({resolved.name}). "
                "Provide compatible weights and set VISION_MODEL_PATH in .env; see models/README.md."
            ),
            "model_path": resolved.name,
        }

    backend = "onnxruntime" if (suffix == ".onnx" and has_ort) else "ultralytics"
    return {
        "ready": True,
        "code": "ready",
        "backend": backend,
        "model_filename": resolved.name,
        "model_name": manifest.get("model_name", resolved.stem),
        "architecture": manifest.get("architecture", "YOLOv11-n"),
        "input_mode": manifest.get("input_mode", "archive_png"),
        "imgsz": int(manifest.get("imgsz", 640)),
        "message": (
            f"Vision meteor detector ready: {manifest.get('model_name', resolved.stem)} "
            f"({backend}, {manifest.get('input_mode', 'archive_png')})."
        ),
    }


def extract_spectrogram_canvas(image) -> tuple[Any, tuple[int, int, int, int]]:
    """Extract the pure time-frequency spectrogram canvas from framed or raw PNGs.

    1. Official BRAMS / PyBRAMS framed PNGs use exact padding (left=64, top=64,
       right=96, bottom=96) bounded by a 1-px [128, 128, 128] border line on a
       black (dark-mode) or white (light-mode) frame. Cropping (64, 64, W-96, H-96)
       recovers the exact borderless Radio Meteor Zoo pixel grid (e.g. 864x595).
    2. Borderless Radio Meteor Zoo PNGs have jet pixels up to their edges and are
       returned unchanged with crop_box=(0, 0, W, H).
    3. Generic white-bordered Matplotlib figures fall back to active-plot cropping.
    """
    import numpy as np

    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    height, width, _ = arr.shape
    if width < 64 or height < 64:
        return rgb, (0, 0, width, height)

    # 1. Exact BIRA-IASB / PyBRAMS C-style frame check
    if width > 200 and height > 200:
        x0, y0, x1, y1 = 64, 64, width - 96, height - 96
        top_border = arr[y0 - 1, x0:x1].astype(np.int16)
        left_border = arr[y0:y1, x0 - 1].astype(np.int16)
        gray_top = float(np.mean(np.all(np.abs(top_border - 128) <= 3, axis=1)))
        gray_left = float(np.mean(np.all(np.abs(left_border - 128) <= 3, axis=1)))
        if gray_top > 0.85 and gray_left > 0.85:
            return rgb.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

    border = np.concatenate([
        arr[:2, :, :].reshape(-1, 3),
        arr[-2:, :, :].reshape(-1, 3),
        arr[:, :2, :].reshape(-1, 3),
        arr[:, -2:, :].reshape(-1, 3),
    ], axis=0)
    # Borderless spectrograms do not have a uniform white margin.
    if float(border.mean()) < 225.0 or float(border.std()) > 35.0:
        return rgb, (0, 0, width, height)

    # Identify pixels belonging to saturated/dark plot regions (not white paper or thin grey text).
    channel_spread = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    brightness = arr.mean(axis=2)
    is_plot_pixel = (channel_spread > 25) | (brightness < 180.0)

    row_density = is_plot_pixel.mean(axis=1)
    active_rows = np.where(row_density > 0.35)[0]
    if active_rows.size < int(0.40 * height):
        return rgb, (0, 0, width, height)
    y0, y1 = int(active_rows[0]), int(active_rows[-1]) + 1

    col_density = is_plot_pixel[y0:y1, :].mean(axis=0)
    active_cols = np.where(col_density > 0.55)[0]
    if active_cols.size < int(0.40 * width):
        return rgb, (0, 0, width, height)

    # Split contiguous horizontal runs so a right-side colorbar separated by a
    # white margin is excluded in favor of the widest main spectrogram axis.
    gaps = np.where(np.diff(active_cols) > max(4, int(0.01 * width)))[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, active_cols.size - 1]
    best_idx = int(np.argmax(active_cols[ends] - active_cols[starts]))
    x0 = int(active_cols[starts[best_idx]])
    x1 = int(active_cols[ends[best_idx]]) + 1

    if (x1 - x0) < int(0.40 * width) or (y1 - y0) < int(0.40 * height):
        return rgb, (0, 0, width, height)
    return rgb.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def render_rmz_spectrogram_bytes(
    samples,
    sample_rate_hz: float,
    *,
    carrier_hz: float | None = None,
    width: int = RMZ_WIDTH,
    height: int = RMZ_HEIGHT,
    band_hz: float = RMZ_BAND_HZ,
) -> tuple[bytes, dict[str, Any]]:
    """Render a borderless 864x595 RGB jet spectrogram matching Radio Meteor Zoo."""
    import matplotlib
    import numpy as np
    from PIL import Image
    from scipy import signal as sp_signal

    data = np.asarray(samples, dtype=np.float64)
    if data.ndim > 1:
        data = data[:, 0]
    if data.size < 64 or not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Invalid waveform for RMZ spectrogram rendering.")

    nperseg = min(RMZ_FFT_SIZE, max(64, 1 << int(math.floor(math.log2(data.size // 2 or 64)))))
    noverlap = min(nperseg - 1, int(round(RMZ_OVERLAP_RATIO * nperseg)))
    freqs, _, power = sp_signal.spectrogram(
        data,
        fs=float(sample_rate_hz),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    half_band = band_hz / 2.0
    nyquist = sample_rate_hz / 2.0
    if carrier_hz is None or not math.isfinite(carrier_hz):
        search_mask = (freqs >= 300.0) & (freqs <= max(400.0, nyquist - half_band))
        if np.any(search_mask):
            median_profile = np.median(power[search_mask, :], axis=1)
            carrier_hz = float(freqs[search_mask][int(np.argmax(median_profile))])
        else:
            carrier_hz = min(1000.0, nyquist / 2.0)

    low_hz = max(0.0, float(carrier_hz) - half_band)
    high_hz = min(nyquist, low_hz + band_hz)
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if np.count_nonzero(band_mask) < 4:
        band_mask = slice(None)
        low_hz, high_hz = float(freqs[0]), float(freqs[-1])

    # Flip frequency rows so higher frequencies appear at the top (origin='lower').
    sub_power = power[band_mask, :][::-1, :]
    db = 10.0 * np.log10(np.maximum(sub_power, 1e-24))

    median_db = float(np.median(db))
    mad_db = float(np.median(np.abs(db - median_db))) * 1.4826
    sigma_db = max(mad_db, 1.0)
    vmin = median_db - 1.5 * sigma_db
    vmax = max(vmin + 28.0, float(np.percentile(db, 99.7)))

    norm = np.clip((db - vmin) / max(1e-6, vmax - vmin), 0.0, 1.0)
    cmap = matplotlib.colormaps["jet"]
    rgb_float = cmap(norm)[..., :3]
    rgb_uint8 = (rgb_float * 255.0 + 0.5).astype(np.uint8)

    pil_img = Image.fromarray(rgb_uint8, mode="RGB").resize(
        (int(width), int(height)), Image.Resampling.BILINEAR
    )
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    meta = {
        "carrier_hz": round(float(carrier_hz), 3),
        "frequency_band_hz": [round(low_hz, 3), round(high_hz, 3)],
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "dimensions": [int(width), int(height)],
        "colormap": "jet",
    }
    return buffer.getvalue(), meta


def render_rmz_from_wav_path(wav_file: Path) -> tuple[bytes, dict[str, Any]]:
    """Render an exact BIRA-IASB / Radio Meteor Zoo borderless jet spectrogram from a BRAMS WAV."""
    import tempfile
    import warnings
    from PIL import Image
    from scipy.io import wavfile

    try:
        from pybrams.brams.wavfile import WavFile
        from pybrams.plot import spectrogram as brams_spectrogram
        from pybrams.signal.processing.signal import Signal

        decoded = WavFile.from_file(wav_file)
        if decoded.series is not None:
            rate = float(decoded.series.samplerate)
            sig = Signal(decoded.series, decoded.pps)
            carrier_hz = float(sig.beacon_frequency)
            with tempfile.TemporaryDirectory(prefix="brams-rmz-") as tmp_dir:
                out_png = Path(tmp_dir) / "rmz.png"
                brams_spectrogram.plot(
                    decoded,
                    central_frequency=carrier_hz,
                    export=True,
                    filename=str(out_png),
                    use_turbo_palette=False,
                    axes=False,
                    half_range_spect=100,
                )
                if out_png.is_file() and out_png.stat().st_size > 0:
                    # Convert 8-bit indexed palette PNG to standard RGB PNG for uniform loading
                    with Image.open(out_png) as pal_img:
                        rgb_img = pal_img.convert("RGB")
                        dims = [rgb_img.width, rgb_img.height]
                        buf = BytesIO()
                        rgb_img.save(buf, format="PNG")
                    return buf.getvalue(), {
                        "carrier_hz": round(carrier_hz, 3),
                        "frequency_band_hz": [round(carrier_hz - 100.0, 3), round(carrier_hz + 100.0, 3)],
                        "dimensions": dims,
                        "colormap": "jet",
                        "renderer": "pybrams.plot.spectrogram",
                    }
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        int_rate, raw = wavfile.read(wav_file)
    return render_rmz_spectrogram_bytes(raw, float(int_rate))


def letterbox_rgb(
    image,
    imgsz: int = 640,
    resize_mode: str = "letterbox",
) -> tuple[Any, dict[str, float]]:
    """Prepare [1, 3, imgsz, imgsz] float32 RGB in [0, 1] matching Ultralytics."""
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    if resize_mode == "stretch":
        resized = rgb.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw), {
            "mode": "stretch",
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
            "scale_x": imgsz / float(orig_w),
            "scale_y": imgsz / float(orig_h),
            "pad_x": 0.0,
            "pad_y": 0.0,
        }

    scale = min(imgsz / float(orig_w), imgsz / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))

    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(resized, (left, top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw), {
        "mode": "letterbox",
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "scale_x": scale,
        "scale_y": scale,
        "pad_x": float(left),
        "pad_y": float(top),
    }


def nms_boxes(boxes_xyxy, scores, iou_threshold: float = 0.45, max_det: int = 200, *, return_truncation: bool = False):
    """Pure-NumPy greedy IoU Non-Maximum Suppression."""
    import numpy as np

    if len(scores) == 0:
        return ([], False) if return_truncation else []
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            order = order[:0]
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-9
        iou = inter / union
        order = rest[iou <= iou_threshold]
    return (keep, bool(order.size)) if return_truncation else keep


class _DetectionList(list):
    def __init__(self, values=(), *, truncated=False):
        super().__init__(values)
        self.truncated = truncated


def decode_yolo_output(
    raw_output,
    transform: dict[str, float],
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_class: int | None = 0,
) -> list[dict[str, Any]]:
    """Decode Ultralytics YOLOv8/YOLOv11 ONNX output (both raw [1,4+C,N] and NMS [1,N,6])."""
    import numpy as np

    pred = np.asarray(raw_output, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2 or pred.size == 0:
        return []

    # Case 1: End-to-end ONNX export with baked-in NMS -> shape [max_det, 6] (x1, y1, x2, y2, score, cls)
    truncated = False
    if pred.shape[1] == 6 and pred.shape[0] <= 1000:
        mask = pred[:, 4] >= conf_threshold
        if target_class is not None:
            mask &= np.round(pred[:, 5]).astype(int) == target_class
        filtered = pred[mask]
        boxes_xyxy = filtered[:, :4].copy()
        scores = filtered[:, 4]
        classes = np.round(filtered[:, 5]).astype(int)
    else:
        # Case 2: Standard Ultralytics anchor-free head -> [4 + nc, 8400] (or transposed [8400, 4 + nc])
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 128:
            pred = pred.T  # -> [8400, 4 + nc]
        if pred.shape[1] < 5:
            return []
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        class_scores = pred[:, 4:]
        if target_class is not None and target_class < class_scores.shape[1]:
            scores = class_scores[:, target_class]
            classes = np.full(scores.shape, target_class, dtype=int)
        else:
            classes = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(class_scores.shape[0]), classes]

        mask = scores >= conf_threshold
        if not np.any(mask):
            return []
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        scores = scores[mask]
        classes = classes[mask]
        boxes_xyxy = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        keep, truncated = nms_boxes(boxes_xyxy, scores, iou_threshold=iou_threshold, return_truncation=True)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        classes = classes[keep]

    orig_w = transform["orig_w"]
    orig_h = transform["orig_h"]
    pad_x = transform["pad_x"]
    pad_y = transform["pad_y"]
    scale_x = transform["scale_x"]
    scale_y = transform["scale_y"]

    detections: list[dict[str, Any]] = []
    for box, score, cls_id in zip(boxes_xyxy, scores, classes):
        x1 = float(np.clip((box[0] - pad_x) / scale_x, 0.0, orig_w))
        y1 = float(np.clip((box[1] - pad_y) / scale_y, 0.0, orig_h))
        x2 = float(np.clip((box[2] - pad_x) / scale_x, 0.0, orig_w))
        y2 = float(np.clip((box[3] - pad_y) / scale_y, 0.0, orig_h))
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        detections.append({
            "class_id": int(cls_id),
            "confidence": float(score),
            "canvas_xyxy": [x1, y1, x2, y2],
        })
    return _DetectionList(detections, truncated=truncated)


class _OnnxDetector:
    def __init__(self, model_path: Path, manifest: dict[str, Any]):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.manifest = manifest
        self.imgsz = int(manifest.get("imgsz", 640))
        self.resize_mode = str(manifest.get("resize_mode", "letterbox"))
        self.backend = "onnxruntime"

    def predict(self, canvas_image, conf_threshold: float, iou_threshold: float) -> list[dict[str, Any]]:
        tensor, transform = letterbox_rgb(canvas_image, imgsz=self.imgsz, resize_mode=self.resize_mode)
        outputs = self.session.run(None, {self.input_name: tensor})
        return decode_yolo_output(
            outputs[0],
            transform,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_class=0,
        )


class _UltralyticsDetector:
    def __init__(self, model_path: Path, manifest: dict[str, Any]):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="detect")
        self.manifest = manifest
        self.imgsz = int(manifest.get("imgsz", 640))
        self.backend = "ultralytics"

    def predict(self, canvas_image, conf_threshold: float, iou_threshold: float) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=canvas_image,
            imgsz=self.imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device="cpu",
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        out = []
        for box, score, cls_id in zip(xyxy, confs, clss):
            if int(cls_id) != 0:
                continue
            out.append({
                "class_id": 0,
                "confidence": float(score),
                "canvas_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            })
        return out


def _get_detector(model_path: Path):
    global _CACHED_DETECTOR
    resolved = model_path.resolve()
    mtime_ns = resolved.stat().st_mtime_ns
    key = str(resolved)
    with VISION_LOCK:
        if _CACHED_DETECTOR is not None and _CACHED_DETECTOR[0] == key and _CACHED_DETECTOR[1] == mtime_ns:
            return _CACHED_DETECTOR[2]
        manifest = load_model_manifest(resolved)
        if resolved.suffix.lower() == ".onnx" and find_spec("onnxruntime") is not None:
            detector = _OnnxDetector(resolved, manifest)
        else:
            detector = _UltralyticsDetector(resolved, manifest)
        _CACHED_DETECTOR = (key, mtime_ns, detector)
        return detector


def render_annotated_spectrogram(
    full_image,
    detections: list[dict[str, Any]],
    *,
    header_text: str,
) -> bytes:
    """Draw non-destructive bounding boxes and a summary header bar onto a PNG copy."""
    from PIL import Image, ImageDraw

    base = full_image.convert("RGB")
    banner_h = 28
    out = Image.new("RGB", (base.width, base.height + banner_h), (15, 23, 42))
    out.paste(base, (0, banner_h))
    draw = ImageDraw.Draw(out)
    draw.text((10, 7), header_text[:120], fill=(241, 245, 249))

    for det in detections:
        x1, y1, x2, y2 = det["image_xyxy"]
        y1_b, y2_b = y1 + banner_h, y2 + banner_h
        high = det["confidence"] >= 0.50
        color = (0, 255, 136) if high else (255, 209, 102)
        # Expand 2px outward slightly so ultra-thin 2-3px meteor traces remain visible inside the box!
        bx1 = max(0.0, x1 - 2.0)
        by1 = max(float(banner_h), y1_b - 2.0)
        bx2 = min(float(out.width - 1), x2 + 2.0)
        by2 = min(float(out.height - 1), y2_b + 2.0)
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)

        tag = f"{det['confidence']:.2f}"
        label_y = by1 - 14 if by1 - 14 >= banner_h else min(out.height - 14, by2 + 2)
        label_x = min(max(2.0, bx1), max(2.0, out.width - 36.0))
        draw.rectangle([label_x, label_y, label_x + 32, label_y + 13], fill=(15, 23, 42))
        draw.text((label_x + 3, label_y + 1), tag, fill=color)

    buffer = BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


def create_annotated_overlay(
    png_bytes: bytes,
    detections_with_image_xyxy: list[dict[str, Any]],
    *,
    recording_start: datetime | None = None,
    recording_end: datetime | None = None,
    receiver_label: str = "",
    model_name: str = "yolo11n_brams",
    conf_threshold: float = 0.25,
) -> str:
    """Render and persist an annotated detection PNG, returning its opaque artifact ID."""
    from PIL import Image

    with Image.open(BytesIO(png_bytes)) as opened:
        full_rgb = opened.convert("RGB")
    window_label = ""
    if recording_start is not None and recording_end is not None:
        window_label = f" · {_utc_iso(recording_start)} -> {recording_end.astimezone(timezone.utc).strftime('%H:%M:%SZ')}"
    station_part = receiver_label or "BRAMS"
    verdict = (
        f"{len(detections_with_image_xyxy)} meteor echo(es)"
        if detections_with_image_xyxy
        else "No meteor echoes detected"
    )
    header = f"{station_part}{window_label} · {verdict} ({model_name}, conf>={conf_threshold:g})"
    overlay_bytes = render_annotated_spectrogram(full_rgb, detections_with_image_xyxy, header_text=header)
    return save_detection_image(overlay_bytes)


def analyze_spectrogram_bytes(
    png_bytes: bytes,
    *,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    recording_start: datetime | None = None,
    recording_end: datetime | None = None,
    receiver_label: str = "",
    save_overlay: bool = True,
    include_image_boxes: bool = False,
) -> dict[str, Any]:
    """Run YOLO meteor echo detection on one spectrogram PNG and map to UTC times."""
    from PIL import Image

    started = perf_counter()
    resolved = resolve_vision_model_path(model_path)
    detector = _get_detector(resolved)

    with Image.open(BytesIO(png_bytes)) as opened:
        full_rgb = opened.convert("RGB")

    canvas_rgb, crop_box = extract_spectrogram_canvas(full_rgb)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    canvas_w = max(1, crop_x1 - crop_x0)
    canvas_h = max(1, crop_y1 - crop_y0)

    raw_dets = detector.predict(canvas_rgb, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    # Sort chronologically across the horizontal time axis.
    raw_dets.sort(key=lambda item: (item["canvas_xyxy"][0] + item["canvas_xyxy"][2]) / 2.0)

    duration_s = 300.0
    if recording_start is not None and recording_end is not None and recording_end > recording_start:
        duration_s = (recording_end - recording_start).total_seconds()

    structured: list[dict[str, Any]] = []
    for idx, det in enumerate(raw_dets, start=1):
        cx1, cy1, cx2, cy2 = det["canvas_xyxy"]
        ix1, iy1 = crop_x0 + cx1, crop_y0 + cy1
        ix2, iy2 = crop_x0 + cx2, crop_y0 + cy2
        xc_norm = ((cx1 + cx2) / 2.0) / canvas_w
        yc_norm = ((cy1 + cy2) / 2.0) / canvas_h
        w_px = max(1e-3, cx2 - cx1)
        h_px = max(1e-3, cy2 - cy1)
        t_center_s = xc_norm * duration_s
        t_start_s = (cx1 / canvas_w) * duration_s
        t_end_s = (cx2 / canvas_w) * duration_s

        entry: dict[str, Any] = {
            "echo_index": idx,
            "confidence": round(det["confidence"], 4),
            "confidence_tier": "high" if det["confidence"] >= 0.50 else "moderate",
            "time_offset_seconds": round(t_center_s, 2),
            "time_span_seconds": [round(t_start_s, 2), round(t_end_s, 2)],
            "bbox_norm_xywh": [
                round(xc_norm, 5),
                round(yc_norm, 5),
                round(w_px / canvas_w, 5),
                round(h_px / canvas_h, 5),
            ],
            "aspect_ratio_hw": round(h_px / w_px, 2),
            "area_percent": round(100.0 * (w_px * h_px) / (canvas_w * canvas_h), 4),
            "image_xyxy": [round(ix1, 1), round(iy1, 1), round(ix2, 1), round(iy2, 1)],
        }
        if recording_start is not None:
            entry["estimated_echo_utc"] = _utc_iso(recording_start + timedelta(seconds=t_center_s))
        structured.append(entry)

    model_label = str(detector.manifest.get("model_name", resolved.stem))
    overlay_artifact_id = None
    if save_overlay:
        overlay_artifact_id = create_annotated_overlay(
            png_bytes,
            structured,
            recording_start=recording_start,
            recording_end=recording_end,
            receiver_label=receiver_label,
            model_name=model_label,
            conf_threshold=conf_threshold,
        )

    if include_image_boxes:
        llm_detections = structured
    else:
        # Strip pixel-space helper coordinates from LLM-facing dict to keep context compact.
        llm_detections = [{k: v for k, v in item.items() if k != "image_xyxy"} for item in structured]
    confidences = [item["confidence"] for item in llm_detections]

    return {
        "meteor_detected": bool(llm_detections),
        "meteor_count": len(llm_detections),
        "detection_limit_reached": bool(getattr(raw_dets, "truncated", False)),
        "high_confidence_count": sum(item["confidence_tier"] == "high" for item in llm_detections),
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "detections": llm_detections,
        "annotated_artifact_id": overlay_artifact_id,
        "image_dimensions": [full_rgb.width, full_rgb.height],
        "analyzed_canvas_box": list(crop_box),
        "inference_ms": round((perf_counter() - started) * 1000.0, 2),
        "model_name": model_label,
        "model_architecture": detector.manifest.get("architecture", "YOLOv11-n"),
        "backend": detector.backend,
        "confidence_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
    }
