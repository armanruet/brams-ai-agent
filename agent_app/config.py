"""Configuration belongs to this process, not to the chat conversation."""

from dataclasses import dataclass, field
import os
import math
from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VISION_MODEL = "models/yolo11n_brams.onnx"
DEFAULT_MODEL = "ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M"
DEFAULT_LOCAL_CONTEXT_SIZE = 16384


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = DEFAULT_MODEL
    api_key: str = field(default="local-not-needed", repr=False)
    timeout_seconds: float = 180.0
    max_tokens: int = 4096
    recovery_max_tokens: int = 6144
    context_size: int | None = None
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860
    provider: Literal["local", "openai", "compatible"] = "local"
    token_parameter: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    assistant_scope: Literal["domain", "general"] = "domain"
    vision_model_path: str = DEFAULT_VISION_MODEL
    vision_conf_threshold: float = 0.25
    vision_iou_threshold: float = 0.45
    vision_input_source: Literal["auto", "archive_png", "rmz_from_wav"] = "auto"
    request_limit: int = 6
    tool_calls_limit: int = 6

    def __post_init__(self) -> None:
        if self.provider not in {"local", "openai", "compatible"}:
            raise ValueError("Choose local, openai or compatible as provider.")
        if self.token_parameter not in {"auto", "max_tokens", "max_completion_tokens"}:
            raise ValueError("Invalid token-limit parameter.")
        if self.assistant_scope not in {"domain", "general"}:
            raise ValueError("ASSISTANT_SCOPE must be domain or general.")
        if self.vision_input_source not in {"auto", "archive_png", "rmz_from_wav"}:
            raise ValueError("VISION_INPUT_SOURCE must be auto, archive_png or rmz_from_wav.")
        if not math.isfinite(self.vision_conf_threshold) or not 0.0 < self.vision_conf_threshold <= 1.0:
            raise ValueError("VISION_CONF_THRESHOLD must be in (0, 1].")
        if not math.isfinite(self.vision_iou_threshold) or not 0.0 < self.vision_iou_threshold <= 1.0:
            raise ValueError("VISION_IOU_THRESHOLD must be in (0, 1].")
        if not self.vision_model_path.strip():
            raise ValueError("VISION_MODEL_PATH cannot be empty.")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("API base URL must be an http(s) URL, ending in /v1.")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("Put credentials in the API key field, not the base URL.")
        if self.provider != "compatible" and not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError("API base URL must end in /v1 (not /chat/completions).")
        if self.provider == "compatible" and parsed.path.rstrip("/").endswith(("/chat/completions", "/responses", "/models")):
            raise ValueError("Enter the API base URL, not a /chat/completions, /responses or /models endpoint.")
        if self.provider == "openai" and self.base_url.rstrip("/") != "https://api.openai.com/v1":
            raise ValueError("OpenAI mode uses https://api.openai.com/v1. Choose compatible mode for other endpoints.")
        if not self.model.strip():
            raise ValueError("Model ID cannot be empty; enter an exact ID supported by the provider.")
        if self.timeout_seconds <= 0 or self.max_tokens < 1:
            raise ValueError("Timeout and max tokens must be positive.")
        if self.request_limit < 1 or self.tool_calls_limit < 1:
            raise ValueError("Model request and tool-call limits must be positive.")
        if self.recovery_max_tokens < self.max_tokens:
            raise ValueError("LLAMA_RECOVERY_MAX_TOKENS must be at least LLAMA_MAX_TOKENS.")
        if self.context_size is not None and self.context_size < 1024:
            raise ValueError("LLAMA_CTX_SIZE must be at least 1024 tokens.")
        if not 1 <= self.gradio_port <= 65535:
            raise ValueError("GRADIO_PORT must be between 1 and 65535.")

    @property
    def uses_completion_tokens(self) -> bool:
        return (self.token_parameter == "max_completion_tokens"
                or (self.token_parameter == "auto" and self.provider == "openai"))


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    max_tokens = int(os.getenv("LLAMA_MAX_TOKENS", "4096"))
    raw_ctx = os.getenv("LLAMA_CTX_SIZE", "").strip()
    return Settings(
        base_url=os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/"),
        model=os.getenv("LLAMA_MODEL", DEFAULT_MODEL),
        api_key=os.getenv("LLAMA_API_KEY", "local-not-needed") or "local-not-needed",
        timeout_seconds=float(os.getenv("LLAMA_TIMEOUT_SECONDS", "180")),
        max_tokens=max_tokens,
        recovery_max_tokens=int(os.getenv("LLAMA_RECOVERY_MAX_TOKENS",
                                          str(max(6144, max_tokens)))),
        context_size=int(raw_ctx) if raw_ctx else None,
        assistant_scope=os.getenv("ASSISTANT_SCOPE", "domain").strip().lower(),
        gradio_host=os.getenv("GRADIO_HOST", "127.0.0.1"),
        gradio_port=int(os.getenv("GRADIO_PORT", "7860")),
        vision_model_path=os.getenv("VISION_MODEL_PATH", DEFAULT_VISION_MODEL).strip(),
        vision_conf_threshold=float(os.getenv("VISION_CONF_THRESHOLD", "0.25")),
        vision_iou_threshold=float(os.getenv("VISION_IOU_THRESHOLD", "0.45")),
        vision_input_source=os.getenv("VISION_INPUT_SOURCE", "auto").strip().lower(),
        request_limit=int(os.getenv("LLAMA_REQUEST_LIMIT", "6")),
        tool_calls_limit=int(os.getenv("LLAMA_TOOL_CALL_LIMIT", "6")),
    )
