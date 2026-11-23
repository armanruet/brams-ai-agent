"""Per-browser connection profiles. Secrets stay in Gradio's server-side State."""

from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit
import ipaddress

from .config import Settings


OPENAI_BASE_URL = "https://api.openai.com/v1"
PROVIDER_NAMES = {"local": "Local", "openai": "Public API", "compatible": "Public API"}


@dataclass(frozen=True)
class ConnectionProfiles:
    local: Settings
    active: Settings
    saved_hosted: dict[str, Settings] = field(default_factory=dict, repr=False)

    @classmethod
    def initial(cls, settings: Settings):
        return cls(local=settings, active=settings)


def connection_label(settings: Settings) -> str:
    return f"{PROVIDER_NAMES[settings.provider]} · {settings.model}"


def is_loopback(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        return False


def apply_profile(profiles: ConnectionProfiles, provider: str, base_url: str,
                  model: str, api_key: str, token_parameter: str = "auto") -> ConnectionProfiles:
    """Validate first; return a new profile set without mutating prior sessions.

    Empty key can reuse a key only for the same provider AND exact normalized
    endpoint. Changing a destination never forwards the old destination's key.
    """
    if provider == "local":
        return replace(profiles, active=profiles.local)
    if provider not in {"openai", "compatible"}:
        raise ValueError("Choose a supported provider.")
    base_url = (base_url or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback(parsed.hostname)):
        raise ValueError("Use HTTPS for a hosted API. HTTP is supported only for loopback servers.")
    previous = profiles.saved_hosted.get(provider)
    key = (api_key or "").strip()
    if not key and previous and previous.base_url == base_url:
        key = previous.api_key
    if not key:
        if is_loopback(parsed.hostname):
            key = "local-not-needed"
        else:
            raise ValueError("Enter an API key for this endpoint. Keys are reused only for the same provider and base URL.")
    settings = replace(profiles.local, provider=provider, base_url=base_url, model=(model or "").strip(),
                       api_key=key, token_parameter=token_parameter)
    return replace(profiles, active=settings, saved_hosted={**profiles.saved_hosted, provider: settings})


def redact(value, api_key: str):
    """Keep credentials out of UI diagnostics, including provider error content."""
    if isinstance(value, str):
        return value.replace(api_key, "[redacted]") if api_key else value
    if isinstance(value, list):
        return [redact(item, api_key) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, api_key) for key, item in value.items()}
    return value
