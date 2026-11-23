"""Deterministic anti-hallucination and grounding checks for model responses."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from pydantic_ai.messages import ModelMessage, ToolReturnPart

from .benchmark_audit import scientific_flags
from .recovery import evidence_summary


_BRAMS_DATA_PROMPT_RE = re.compile(
    r"\b("
    r"check\s+if\s+.+\s+is\s+available|"
    r"which\s+(?:receivers?|stations?)\s+(?:are\s+available|have\s+recordings?)|"
    r"give\s+me\s+the\s+exact\s+coordinates|"
    r"coordinates\s+of\s+station|"
    r"show\s+me\s+\d*\s*spectrogram|"
    r"play\s+(?:one|a|the)\s+wav|"
    r"download\s+(?:one|a|the)\s+(?:wav|spectrogram)|"
    r"detect\s+meteor|meteor\s+echo(?:es)?\s+in|"
    r"noise\s+psd|calibrator\s+psd|"
    r"(?:north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?)\s+(?:(?:part|half|region)\s+of\s+|of\s+)?belgium|"
    r"(?-i:[A-Z]{6}(?:_SYS\d{3})?)"
    r")\b",
    re.IGNORECASE,
)

# Prompts that are pure capability, policy, or refusal questions do not require a tool call.
# Phase Next: remove ads-?b|aircraft from _NON_TOOL_EXEMPT_RE and deterministic_limit_response when check_aircraft is registered.
_NON_TOOL_EXEMPT_RE = re.compile(
    r"\b("
    r"delete|clear\s+cache|remove\s+files|ads-?b|aircraft|optical|camera|cams|"
    r"what\s+can\s+you\s+do|capabilities|"
    r"entire\s+archive|all\s+wav\s+files\s+from\s+the\s+entire"
    r")\b",
    re.IGNORECASE,
)

# Availability/active phrasing should only require check_availability when accompanied
# by BRAMS, station, archive, or time-window context (never e.g. "Is Python 3.14 available?").
_AVAILABILITY_CONTEXT_RE = re.compile(
    r"\b(?:brams|stations?|receivers?|sites?|antennas?|archive|archived|recordings?|"
    r"hourly|coverage|data|files?|network|radar|meteor|spectrogram|wav|psd|"
    r"utc|gmt|yesterday|today|tomorrow|from|between|"
    r"belgium|luxembourg|kirchberg|humain|dourbes|bilzen|ecotron|ieper|dinant|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"\d{4}[-/]\d|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}:\d{2}|(?-i:[A-Z]{6}(?:_SYS\d{3})?))\b",
    re.IGNORECASE,
)

# Domain cues protect legitimate BRAMS, radio-meteor, radar, signal-processing, and time questions.
_DOMAIN_CUE_RE = re.compile(
    r"\b(?:brams|pybrams|meteors?|echo(?:es)?|beacons?|transmitters?|receivers?|stations?|antennas?|"
    r"spectrograms?|wavs?|audio|psd|calibrator|noise\s+floor|frequenc(?:y|ies)|k?m?hz|"
    r"doppler|fresnel|forward[-\s]+scatter|radar|ionospher\w*|ablation|sporadic[-\s]+e|"
    r"perseids?|geminids?|quadrantids?|leonids?|orionids?|meteor\s+shower|"
    r"fft|stft|hann|hamming|spectrum|bandwidth|snr|sample\s+rate|nyquist|pps|gps|clock|drift|"
    r"utc|gmt|brussels|belgium|belgian|luxembourg|dourbes|humain|kirchberg|bilzen|ecotron|ieper|dinant|"
    r"archive|archived|availability|coverage|recordings?|datasets?|airplanes?|aircraft|ads-?b|"
    r"optical|cams|gmn|yolo|detect\w*|(?-i:[A-Z]{6}(?:_SYS\d{3})?))\b",
    re.IGNORECASE,
)

# Narrow, high-confidence off-domain intents (paired with zero domain cues — never refuse on cue absence alone).
_HIGH_CONFIDENCE_OFF_DOMAIN_RE = re.compile(
    r"\b(?:"
    r"(?:write|compose|tell|create)\s+(?:me\s+)?(?:a\s+)?(?:short\s+|funny\s+)?(?:poem|haiku|story|joke|song|lyrics|essay|limerick|fairy\s+tale|fanfiction|cover\s+letter|resume)|"
    r"recipe\s+for|how\s+to\s+(?:cook|bake|make)\s+(?:a\s+)?(?:cake|pasta|pizza|soup|bread|curry|cookies|pancakes|omelette|dinner|breakfast|lunch)|ingredients\s+for|"
    r"what(?:'s|\s+is)\s+the\s+capital\s+of|population\s+of\s+[a-z]|who\s+(?:won\s+the\s+(?:world\s+cup|super\s+bowl|election|match|game)|is\s+the\s+(?:president|prime\s+minister|king|queen|ceo)\s+of)|"
    r"(?:write|give|show)\s+(?:me\s+)?(?:a\s+)?(?:python|javascript|java|c\+\+|rust|sql|html|css|bash)\s+(?:script|code|function|program)|"
    r"is\s+(?:python|node(?:\.js)?|java|ios|android|windows|macos)\s+\d|"
    r"best\s+(?:restaurant|hotel|movie|book|laptop|phone)|book\s+(?:a\s+)?(?:flight|hotel|table)|tickets?\s+available|"
    r"(?:medical|legal|tax|investment|dating|relationship)\s+advice|diagnose\s+my\s+symptoms|stock\s+price\s+of"
    r")\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_MESSAGE = (
    "I’m a domain-scoped assistant for the BRAMS radio-meteor network and radio-meteor science. "
    "I can’t help with unrelated general requests, but I can check BRAMS receivers, archive availability, "
    "spectrograms, WAV headers, PSDs, meteor echo detections, or radio-meteor concepts."
)

# Claims that a lookup returned zero/none/no data when the tool actually failed or returned unknown.
_FALSE_ZERO_CLAIM_RE = re.compile(
    r"\b("
    r"no\s+recordings?\s+(?:were\s+|are\s+)?(?:found|available|present|listed)|"
    r"0\s+recordings?|"
    r"zero\s+recordings?|"
    r"no\s+meteor(?:s|\s+echo(?:es)?)\s+(?:were\s+|are\s+)?detected|"
    r"0\s+meteor(?:s|\s+echo(?:es)?)|"
    r"no\s+receivers?\s+(?:were\s+|are\s+)?(?:found|available)|"
    r"0\s+receivers?|"
    r"station\s+was\s+(?:offline|inactive|down)|"
    r"coverage\s+is\s+(?:0%|none)"
    r")\b",
    re.IGNORECASE,
)

_UNKNOWN_QUALIFIER_RE = re.compile(
    r"\b(unknown|failed|error|could\s+not|unable\s+to|cannot\s+determine|setup_required|not\s+evidence)\b",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s)\]\"'`>]+")
_RECEIVER_ID_RE = re.compile(r"\b([A-Z]{6}_SYS\d{3})\b")


def _collect_strings(obj: Any, out: set[str]) -> None:
    if isinstance(obj, str):
        out.add(obj)
        for match in _URL_RE.findall(obj):
            out.add(match.rstrip(".,;:"))
        for match in _RECEIVER_ID_RE.findall(obj):
            out.add(match)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                out.add(k)
            _collect_strings(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _collect_strings(item, out)


def known_evidence_tokens(trace: list[dict[str, Any]], history: Iterable[ModelMessage] = ()) -> tuple[set[str], set[str]]:
    """Only completed tool results can establish sources and receiver IDs."""
    strings: set[str] = set()
    for event in trace:
        if event.get("event") == "tool_result":
            _collect_strings(event.get("result"), strings)
    for message in history:
        for part in getattr(message, "parts", ()):
            if isinstance(part, ToolReturnPart):
                _collect_strings(part.content, strings)
    urls = {s.rstrip(".,;:") for s in strings if s.startswith(("http://", "https://"))}
    receiver_ids = {m for s in strings for m in _RECEIVER_ID_RE.findall(s)}
    return urls, receiver_ids


def _general_meteor_question(prompt: str) -> bool:
    """Recognize explanations of a concept, not questions about an observed result."""
    if not re.search(r"\b(?:meteors?|echo(?:es)?|detect\w*)\b", prompt, re.I):
        return False
    explanation = re.search(
        r"\b(?:what\s+(?:is|are|does|do|causes?)|how\s+(?:does|do|is|are|can)|"
        r"why\s+(?:does|do|is|are)|define|"
        r"(?:explain|describe|tell\s+me)\s+(?:what|how|why))\b", prompt, re.I,
    )
    if not explanation:
        return False
    # References to a specific recording, result, receiver, or time are not
    # general knowledge, even when phrased as a request for an explanation.
    scoped = re.search(
        r"\b(?:this|that|these|those|my|our|result|results|detected|"
        r"station|receiver|yesterday|today|tomorrow)\b|"
        r"\b\d{4}[-/]\d|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{1,2}:\d{2}\b|"
        r"\b[A-Z]{6}_SYS\d{3}\b|\b(?:at|from|for)\s+[A-Z]{6}\b", prompt,
        # Keep the receiver-code alternative case-sensitive.
    )
    # The other alternatives should also recognize mixed-case user prompts.
    scoped = scoped or re.search(
        r"\b(?:this|that|these|those|my|our|result|results|detected|"
        r"station|receiver|yesterday|today|tomorrow)\b", prompt, re.I,
    )
    scoped = scoped or re.search(
        r"\b(?:and|also|then)\s+(?:show|check|find|detect|scan|inspect|"
        r"analy[sz]e|list|download|play|how|which|what)\b", prompt, re.I,
    )
    return not bool(scoped)


def _implicit_detection_followup(prompt: str) -> bool:
    """Recognize detail requests referring to the preceding detection result."""
    detail = re.search(r"\b(?:list|show|give|provide|tell|what|when|which|how\s+many|"
                       r"count|times?|confidence|details?)\b", prompt, re.I)
    subject = re.search(r"\b(?:detected\s+(?:meteor\s+)?echo(?:es)?|"
                        r"detected\s+meteors?|detection\s+(?:results?|details?)|"
                        r"(?:meteor\s+)?echo(?:es)?\s+(?:times?|confidence))\b", prompt, re.I)
    # A named place without a demonstrative is a new scope, not an implicit
    # reference to the previous recording.
    named_place = re.search(r"\b(?:at|from|for|in)\s+(?!(?:the|this|that|these|those)\b)"
                            r"[A-Z][A-Za-zÀ-ÿ'-]+\b", prompt)
    return bool(detail and subject and not named_place)


def _latest_data_result(history: Iterable[ModelMessage]) -> ToolReturnPart | None:
    """Return the latest data result, including failures, rather than older successes."""
    data_tools = {"detect_meteors", "get_spectrogram", "get_wav", "get_psd", "get_recordings",
                  "check_availability", "query_geographic_receivers", "list_receivers", "get_beacon"}
    for message in reversed(list(history)):
        for part in reversed(getattr(message, "parts", ())):
            if isinstance(part, ToolReturnPart) and part.tool_name in data_tools:
                return part
    return None


def _strip_negated_archive_directives(text: str) -> str:
    """Ignore negated instructions such as 'Do not query recordings.' when inferring domains."""
    return re.sub(
        r"\b(?:do\s+not|don['’]?t|never|without)\s+"
        r"(?:query(?:ing)?|check(?:ing)?|fetch(?:ing)?|call(?:ing)?|run(?:ning)?|"
        r"list(?:ing)?|inspect(?:ing)?|access(?:ing)?|us(?:e|ing))\s+"
        r"(?:any\s+|the\s+)?(?:archive\s+|brams\s+)?"
        r"(?:recordings?|availability|stations?|receivers?|archive|tools?)"
        r"(?:\s+(?:or|and)\s+(?:(?:check|query|fetch|list|use)\s+)?(?:any\s+)?"
        r"(?:recordings?|availability|stations?|receivers?|archive|tools?))*\b",
        " ",
        text,
        flags=re.I,
    )


def required_domains(prompt: str) -> set[str]:
    """Recognize bounded operations; do not treat clock calls as archive evidence."""
    parts = re.split(
        r"\b(?:and|also|then)\s+(?=(?:show|check|find|detect|scan|inspect|"
        r"analy[sz]e|list|download|play|how\s+many|which|where)\b)", prompt, maxsplit=1, flags=re.I,
    )
    if len(parts) > 1:
        return required_domains(parts[0]) | required_domains(parts[1])
    if _general_meteor_question(prompt):
        return set()
    text = _strip_negated_archive_directives(prompt.casefold())
    text = re.sub(r"\b(?:do\s+not|don['’]?t|never|without)\s+"
                  r"(?:download(?:ing)?|play(?:ing)?|fetch(?:ing)?)\s+"
                  r"(?:the\s+|any\s+)?(?:wav|audio)\b", " ", text)
    domains = set()
    geo = bool(re.search(r"\b(?:north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?)\b.*\bbelg(?:ium|ian)\b", text))
    meteor_subject = re.search(r"\b(?:meteors?|echo(?:es)?|detect\w*)\b", text)
    detection_action = re.search(
        r"\b(?:detect\w*|find|check|scan|inspect|analy[sz]e|identify|show|list|count|"
        r"how\s+many|what|when|which|any|(?:was|were|is|are)\s+there|does\s+.+\s+contain)\b", text,
    )
    detection = bool(meteor_subject and detection_action and not _general_meteor_question(prompt))
    if geo:
        domains.add("query_geographic_receivers")
    if detection:
        domains.add("detect_meteors")
    if "spectrogram" in text and not detection and not geo:
        domains.add("get_spectrogram")
    psd = bool(re.search(r"\b(?:psd|calibrator|noise level)\b", text))
    recording_listing = bool(re.search(
        r"\brecording\s+metadata\b|\b(?:list|show|find|get|count)\s+"
        r"(?:the\s+|all\s+)?(?:[a-z]{6}(?:_sys\d{3})?\s+)?recordings?\b|"
        r"\b(?:what|which|how\s+many)\s+(?:recordings?|files)\b", text)
        or (re.search(r"\brecordings?\b", text) and re.search(
            r"\b(?:sample\s+rates?|bytes?|size|large|duration|hardware)\b", text)))
    header_request = bool(re.search(r"\brecordings?\b", text) and re.search(
        r"\b(?:header|gps|pps|clock\s+(?:timing|health|sync)|listen\s+at|lo\s*,?\s*beacon)\b", text))
    original_recording = bool(re.search(r"\boriginal\b[^.!?]*\brecording\b", text))
    if (re.search(r"\b(?:wav|audio|playback)\b", text) or header_request or original_recording) and not psd:
        domains.add("get_wav")
    if psd:
        domains.add("get_psd")
    if recording_listing and not (geo or detection or psd or {"get_wav", "get_spectrogram"} & domains):
        domains.add("get_recordings")
    prior_header = bool(re.search(r"\b(?:that|this|returned|previous|earlier)\s+(?:returned\s+)?header\b", text))
    if prior_header:
        domains.add("get_wav")
    if re.search(r"\b(?:beacon|transmitter)\b", text) and not psd and not prior_header and not header_request:
        domains.add("get_beacon")
    if not geo and re.search(r"\b(?:availability|available|active|recordings?)\b", text):
        has_archive_context = bool(
            re.search(r"\b(?:availability|recordings?)\b", text)
            or _AVAILABILITY_CONTEXT_RE.search(prompt)
        )
        if re.search(r"\b(?:which|list)\b.*\breceivers?\b", text) and not re.search(r"\b(?:yesterday|today|from|between|\d{4})\b", text):
            domains.add("list_receivers")
        elif has_archive_context and not ({"get_wav", "get_spectrogram", "detect_meteors", "get_psd", "get_recordings"} & domains):
            domains.add("check_availability")
    if not geo and re.search(r"\b(?:how many|count|number of)\b.*\b(?:stations?|receivers?)\b", text):
        domains.add("list_receivers")
    if (re.search(r"\b(?:where\s+(?:is|are)|(?:latitude|longitude|coordinates?)\s+(?:of|for))\b", text)
            and ("get_beacon" not in domains or re.search(r"\b(?:stations?|receivers?|sites?|antennas?)\b", text))):
        domains.add("list_receivers")
    return domains


def safe_unfulfilled_response(prompt: str, answer: str) -> bool:
    """Allow a narrow clarification or refusal, never an unsupported data answer."""
    if re.search(r"\b(?:and|also|then)\s+(?:show|check|find|detect|scan|inspect|list|download|play)\b", prompt, re.I):
        return False
    if re.search(r"\b(?:detected|found|downloaded|confirmed|measured|processed|cleaned|matched)\b|"
                 r"\b(?:no|zero)\s+(?:available\s+)?(?:recordings?|rows?|meteors?|echoes|data)\b|"
                 r"\b(?:psd|coverage|availability|sample\s+rate)\s+(?:is|was|equals?)\s+(?:\d+(?:\.\d+)?|zero|none)\b|"
                 r"\b\d+(?:\.\d+)?\s*(?:%|hz|db|echoes|recordings)\b", answer, re.I):
        return False
    missing_date = not re.search(r"\b(?:yesterday|today|tomorrow|last\s+\w+|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)\b", prompt, re.I)
    if missing_date and "?" in answer and re.search(r"\b(?:date|day)\b", answer, re.I):
        return True
    bounded_or_unsupported = re.search(r"\b(?:optical|airplanes?|aircraft|denoise|clean\w*|whole\s+network|every\s+recording|all\s+\d+\s+\w*\s*files|every\s+hour|all\s+receivers.*\b\d{4})\b", prompt, re.I)
    refusal = re.search(r"\b(?:cannot|can['’]t|unable|not\s+(?:supported|implemented|available|able)|outside\s+(?:the\s+)?(?:scope|limit)|too\s+(?:large|many)|limit(?:ed|s)?)\b", answer, re.I)
    return bool(bounded_or_unsupported and refusal)


def deterministic_limit_response(
    prompt: str,
    *,
    has_history: bool = False,
    assistant_scope: str = "domain",
    enable_tools: bool = True,
) -> str | None:
    """Handle clearly unfulfillable or high-confidence off-domain requests before calling the model.

    Compound requests are left to the normal agent so their supported parts
    still require evidence. Relative or inherited dates must not be guessed.
    """
    if re.search(r"\b(?:and|also|then)\s+(?:show|check|find|detect|scan|inspect|analy[sz]e|list|download|play|get|what|how|tell|give|provide|compare|count|calculate|explain|describe|summari[sz]e)\b", prompt, re.I):
        return None
    text = prompt.casefold()
    if enable_tools:
        if re.search(r"\b(?:optical|camera|cams)\b", text) and re.search(r"\b(?:match|observation|echo|meteor)\b", text):
            return "I can’t check optical observations or match them to radio echoes. I can show a radio spectrogram or run meteor echo detection for a specified receiver and interval."
        # Phase Next: remove this aircraft refusal branch once check_aircraft is registered.
        if re.search(r"\b(?:airplanes?|aircraft|ads-?b)\b", text) and re.search(r"\b(?:affect\w*|interference|caus\w*|match\w*)\b", text):
            return "I can’t determine whether aircraft affected that recording with the currently available tools. A spectrogram can be retrieved for inspection, but it would not confirm the cause of interference."
        if re.search(r"\b(?:clean|denoise|filter)\b", text) and re.search(r"\brecording\b", text):
            return "I can’t clean or denoise recordings with the currently available tools. I can retrieve the original recording or its spectrogram."
        if (re.search(r"\b(?:whole|entire)\s+network\b.*\b(?:recordings?|files?)\b|\bevery\s+recording\b.*\b(?:network|receivers?)\b", text)
                or re.search(r"\b(?:render|show|create)\b.*\ball\s+\d+\b.*\b(?:files?|spectrograms?)\b", text)
                or re.search(r"\bevery\s+hour\s+of\s+\d{4}\b", text)
                or re.search(r"\ball\s+receivers\b.*\b\d{4}\s+to\s+now\b", text)):
            return "That request exceeds the supported per-request scope. Please choose a receiver and a shorter time interval; I can then check the available data or process a bounded selection."
        if (not has_history and re.search(r"\bpsd\b", text) and re.search(r"\b\d{1,2}:\d{2}\b", text)
                and not re.search(r"\b(?:yesterday|today|tomorrow|last\s+\w+|\d{4})\b", text)
                and not _textual_requested_date_matches(prompt)):
            if re.search(r"\breceiver\s+bedour\b", text):
                return "BEDOUR identifies the BRAMS beacon, rather than a receiver. Which receiver and calendar date should I use for the PSD?"
            return "What calendar date should I use? Please also confirm the receiver ID so I can look up the correct recording."
    if assistant_scope == "domain" and not _DOMAIN_CUE_RE.search(prompt):
        referential = bool(has_history and re.search(r"\b(?:that|this|those|these|it|them|same|above|previous|earlier)\b", text))
        if not referential and _HIGH_CONFIDENCE_OFF_DOMAIN_RE.search(prompt):
            return _OUT_OF_SCOPE_MESSAGE
    return None


_GENERIC_STATION_WORDS = {
    "the", "a", "an", "any", "this", "that", "these", "those", "which", "each", "every", "many",
    "brams", "belgian", "belgium", "radio", "local", "available", "active", "northern", "southern",
    "eastern", "western", "north", "south", "east", "west", "for", "at", "from", "and", "one",
    "other", "yesterday", "today", "tomorrow", "recording", "recordings", "spectrogram", "image",
    "meteor", "meteors", "echo", "echoes", "detection", "wav", "audio", "station", "receiver",
    "result", "results", "hour", "minute", "me", "us", "utc", "gmt", "all", "its", "it",
    "site", "sites", "count", "counts", "unimplemented", "beacon", "transmitter",
    "current", "previous", "earlier", "next", "last", "same", "selected", "requested", "given",
    "archived", "scientific", "around", "about", "approximately", "of", "on", "in", "by", "to", "is", "are", "with", "named", "called",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "fixture", "benchmark", "dataset",
    "saturday", "sunday", "time", "date",
    # Common words, units and labels in multi-part archive requests are not
    # station names merely because they follow "in", "at" or "receiver".
    "example", "most", "catalog", "ids", "decimal", "metres", "meters", "hz", "km",
    "seconds", "minutes", "sample", "byte", "bytes", "registry", "find", "their", "between",
}
_OFFSET_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})", re.I)
_MONTH_NUMBERS = {name: number for number, names in enumerate((
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
    ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
    ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
    ("december", "dec")), start=1) for name in names}
_GENERIC_STATION_WORDS.update(_MONTH_NUMBERS)
_MONTH_PATTERN = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))


def _textual_requested_date_matches(prompt: str) -> list[tuple[int, str]]:
    """Parse explicit English dates and reject impossible calendar dates."""
    results = []
    for match in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\.?\s*,?\s+(\d{{4}})\b", prompt, re.I
    ):
        day, month, year = match.groups()
        results.append((match.start(), datetime(int(year), _MONTH_NUMBERS[month.casefold()], int(day)).date().isoformat()))
    for match in re.finditer(
        rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(\d{{4}})\b", prompt, re.I
    ):
        month, day, year = match.groups()
        results.append((match.start(), datetime(int(year), _MONTH_NUMBERS[month.casefold()], int(day)).date().isoformat()))
    return sorted(results)


def _textual_requested_dates(prompt: str) -> list[str]:
    return [value for _, value in _textual_requested_date_matches(prompt)]


def _receiver_identity_ids(data: dict) -> set[str]:
    """Identify receivers from structured identity fields, never URLs or prose."""
    candidates = [data.get("receiver_id")]
    rows = data.get("receivers")
    if isinstance(rows, list):
        candidates.extend(row.get("receiver_id") for row in rows if isinstance(row, dict))
    # An unsuccessful attempt may only carry its exact requested query. This
    # establishes which lookup failed, never a successful recording identity.
    if data.get("status") not in {None, "ok"}:
        candidates.append(data.get("query"))
    return {candidate.upper() for candidate in candidates if isinstance(candidate, str)
            and _RECEIVER_ID_RE.fullmatch(candidate.upper())}


def _explicit_requested_zone(prompt: str):
    """Resolve an expressed zone for comparing clock labels, without guessing."""
    offset = re.search(r"\b(?:UTC|GMT)\s*([+-])(\d{1,2})(?::?(\d{2}))?\b", prompt, re.I)
    if offset:
        hours, minutes = int(offset.group(2)), int(offset.group(3) or 0)
        if hours > 23 or minutes > 59:
            return None
        seconds = (hours * 60 + minutes) * (1 if offset.group(1) == "+" else -1)
        return timezone(timedelta(minutes=seconds))
    zone = re.search(r"\b([A-Za-z_]+(?:/[A-Za-z_+-]+)+)\b", prompt)
    if not zone:
        zone = re.search(r"\b(CET|CEST|Brussels(?:\s+time)?|Luxembourg\s+time)\b", prompt, re.I)
    if zone:
        from .tools import _resolve_timezone
        name = zone.group(1)
        if name.casefold() == "luxembourg time":
            name = "Europe/Luxembourg"
        try:
            return _resolve_timezone(name)[0]
        except Exception:
            return None
    return timezone.utc


def _requested_clock_range(prompt: str):
    """Return clock labels for explicitly stated ranges, including '10-11 pm'."""
    pattern = (r"\b(?:from|between)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*"
               r"(?:to|and|[-–—])\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b")
    match = re.search(pattern, prompt, re.I)
    if not match:
        # A bare '10–11 pm' or '22:00–23:00' is still an explicit range.
        # Requiring a colon or meridiem avoids mistaking ISO dates for times.
        bare = (r"(?<![:\d])\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|[-–—])\s*"
                r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b(?![/\d:])")
        match = next((item for item in re.finditer(bare, prompt, re.I)
                      if any(item.group(index) for index in (2, 3, 5, 6))), None)
    if not match:
        return None
    first_h, first_m, first_ap, last_h, last_m, last_ap = match.groups()
    if not first_m and not last_m and not first_ap and not last_ap and int(first_h) <= 12 and int(last_h) <= 12:
        return ()  # Both numbers may be AM or PM; do not silently guess.
    if first_ap and not last_ap:
        return ()  # A missing end meridiem may cross midnight; do not guess.
    # A shared trailing am/pm applies to both bounds, e.g. 10-11 pm.
    first_ap = first_ap or last_ap
    result = []
    for hour, minute, meridiem in ((first_h, first_m, first_ap), (last_h, last_m, last_ap)):
        hour, minute = int(hour), int(minute or 0)
        if minute > 59 or (meridiem and not 1 <= hour <= 12) or (not meridiem and hour > 23):
            return ()
        if meridiem:
            hour = hour % 12 + (12 if meridiem.casefold() == "pm" else 0)
        result.append((hour, minute))
    return tuple(result)


def _requested_receiver_names(prompt: str) -> list[str]:
    # A named beacon is an emitter, not a receiver named after the preposition.
    prompt = re.sub(r"\b(from|for|at|in)\s+(?:the\s+)?[A-Za-zÀ-ÿ][\wÀ-ÿ'’-]*\s+(?=beacon\b|transmitter\b)", r"\1 the ", prompt, flags=re.I)
    names_requested = [m.group(1) for m in re.finditer(
        r"\b([A-Za-zÀ-ÿ][\wÀ-ÿ'’-]+(?:\s+\d+(?![:/\d]))?)\s+(?:station|receiver)\b", prompt, re.I)]
    names_requested += [m.group(1) for m in re.finditer(
        r"\b(?:for|at|from|in)\s+(?:the\s+)?([A-Za-zÀ-ÿ][\wÀ-ÿ'’-]+(?:\s+\d+(?![:/\d]))?)\b", prompt, re.I)]
    names_requested += [m.group(1) for m in re.finditer(
        r"\b(?:station|receiver)(?:\s+(?:code|id))?\s+([A-Za-zÀ-ÿ][\wÀ-ÿ'’-]+(?:\s+\d+(?![:/\d]))?)\b", prompt, re.I)]
    for match in re.finditer(
        r"\b(?:for|at|from|in|of|between)\s+(?:the\s+)?([A-Za-zÀ-ÿ][\wÀ-ÿ'’-]+(?:\s+\d+(?![:/\d]))?)\s+"
        r"(?:and|versus|vs\.?)\s+([A-Za-zÀ-ÿ][\wÀ-ÿ'’-]+(?:\s+\d+(?![:/\d]))?)\b", prompt, re.I
    ):
        names_requested.extend(match.groups())
    names_requested = [re.sub(r"['’]s$", "", name, flags=re.I) for name in names_requested]
    return list(dict.fromkeys(name.casefold() for name in names_requested
                              if name.split()[0].casefold() not in _GENERIC_STATION_WORDS
                              and name.casefold() not in {"show", "display", "detect", "check", "play", "list", "where", "how", "what"}
                              and not _RECEIVER_ID_RE.fullmatch(name.upper())))


def _identity_name_text(data: dict) -> str:
    receivers = data.get("receivers") if isinstance(data.get("receivers"), list) else []
    receiver_names = [str(row.get("name", "")) for row in receivers if isinstance(row, dict) and row.get("name")]
    actual_names = " ".join([str(data.get("receiver_name", "")), *receiver_names]).casefold().strip()
    ids = [str(data.get("receiver_id", "")),
           *(str(row.get("receiver_id", "")) for row in receivers if isinstance(row, dict))]
    # Query fallback is needed when catalog rows expose only IDs; if actual names
    # are present they take precedence over potentially incorrect query wording.
    return " ".join([actual_names or str(data.get("query", "")).casefold(), *ids]).casefold().strip()


def _name_in_identity(name: str, text: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(name) + r"(?:(?!\w)|(?=_SYS\d{3}\b))", text, re.I))


def _matches_explicit_scope(prompt: str, data: dict, *, allow_partial_receivers: bool = False) -> bool:
    """Reject evidence that demonstrably concerns a different named receiver or date."""
    emitter = re.search(r"\b([A-Z]{6})\s+(?:beacon|transmitter)\b", prompt, re.I)
    if emitter and data.get("emitter_code") and str(data["emitter_code"]).casefold() != emitter.group(1).casefold():
        return False
    receiver_ids = set(_RECEIVER_ID_RE.findall(prompt.upper()))
    if receiver_ids:
        evidence_ids = _receiver_identity_ids(data)
        primary = data.get("receiver_id")
        if isinstance(primary, str) and _RECEIVER_ID_RE.fullmatch(primary.upper()) and primary.upper() not in receiver_ids:
            return False
        if not receiver_ids.intersection(evidence_ids) or (not allow_partial_receivers and not receiver_ids.issubset(evidence_ids)):
            return False
    names_requested = _requested_receiver_names(prompt)
    if names_requested:
        # Prefer actual returned receiver names/IDs over the query text, which
        # alone does not prove that the returned recording matches the query.
        identity = _identity_name_text(data)
        matches = [_name_in_identity(name, identity) for name in names_requested]
        if not any(matches) or (not allow_partial_receivers and not all(matches)):
            return False

    start_raw, end_raw = data.get("start_utc"), data.get("end_utc")
    if start_raw and end_raw:
        try:
            actual = [datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                      for value in (start_raw, end_raw)]
            if any(value.tzinfo is None for value in actual):
                return False
            actual = [value.astimezone(timezone.utc) for value in actual]
            # A hyphenated UTC range after a date ("2023-12-14 21:15-21:20 UTC")
            # resembles an ISO timestamp with a negative offset. The trailing
            # UTC makes it a clock range, so do not parse 21:20 as an offset.
            iso = [match.group() for match in _OFFSET_ISO.finditer(prompt)
                   if not (re.search(r"\d{2}:\d{2}-\d{2}:\d{2}$", match.group())
                           and re.match(r"\s+(?:UTC|GMT)\b", prompt[match.end():], re.I))]
            if iso:
                requested_local = [datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
                                   for value in iso[:2]]
                requested = [value.astimezone(timezone.utc) for value in requested_local]
                dates = _textual_requested_dates(prompt)
                return (all(actual[i] == value for i, value in enumerate(requested))
                        and all(value in {item.date().isoformat() for item in requested_local} for value in dates))
            zone = _explicit_requested_zone(prompt)
            if zone is None:
                return False
            local = [value.astimezone(zone) for value in actual]
            date_matches = [(match.start(), match.group()) for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", prompt)]
            for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", prompt):
                d, m, y = match.groups()
                date_matches.append((match.start(), f"{y}-{m.zfill(2)}-{d.zfill(2)}"))
            date_matches += _textual_requested_date_matches(prompt)
            dates = [value for _, value in sorted(date_matches)]
            local_dates = [value.date().isoformat() for value in local]
            # A single requested calendar date describes the interval start,
            # including when the end crosses midnight into the following day.
            if dates and dates[0] != local_dates[0]:
                return False
            if len(dates) > 1 and any(value not in local_dates for value in dates[1:]):
                return False
            clocks = _requested_clock_range(prompt)
            if clocks == ():
                return False
            if clocks is None:
                clean = re.sub(r"\b(?:UTC|GMT)\s*[+-]\d{1,2}(?::?\d{2})?\b", "", prompt, flags=re.I)
                found = re.findall(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\b", clean, re.I)
                clocks = []
                for hour, minute, meridiem in found[:2]:
                    hour, minute = int(hour), int(minute)
                    if minute > 59 or (meridiem and not 1 <= hour <= 12) or (not meridiem and hour > 23):
                        return False
                    if meridiem:
                        hour = hour % 12 + (12 if meridiem.casefold() == "pm" else 0)
                    clocks.append((hour, minute))
            if clocks and any((local[i].hour, local[i].minute) != clock for i, clock in enumerate(clocks)):
                return False
        except (ValueError, TypeError, AttributeError, OverflowError):
            return False
    return True


def _matches_geographic_action(prompt: str, data: dict) -> bool:
    if data.get("status") != "ok":
        return True  # A failed attempt is still evidence of an unknown result.
    direction = re.search(r"\b(north|south|east|west)(?:ern)?\b", prompt, re.I)
    if direction and data.get("direction") != direction.group(1).lower():
        return False
    if re.search(r"\bspectrogram\b", prompt, re.I):
        return data.get("action") == "spectrogram"
    if re.search(r"\b(?:active|available|recordings?)\b", prompt, re.I):
        return data.get("action") == "active_count"
    if re.search(r"\b(?:how many|count|number of)\b", prompt, re.I):
        return data.get("action") == "count"
    return True


def requires_tool_or_history(prompt: str, trace: list[dict[str, Any]], history: Iterable[ModelMessage] = ()) -> bool:
    """Require evidence for each recognized operation, avoiding stale history for new requests."""
    needed = required_domains(prompt)
    if not needed:
        if _general_meteor_question(prompt):
            return False
        if _NON_TOOL_EXEMPT_RE.search(prompt) or re.fullmatch(r"\s*(?:hi|hello|help)[!.?\s]*", prompt, re.I):
            return False
        return bool(_BRAMS_DATA_PROMPT_RE.search(prompt)) and not any(
            event.get("event") == "tool_result" and event.get("name") not in {"get_current_utc_time", "convert_local_time", "inspect_time_window"}
            for event in trace)
    needs_coordinates = bool(re.search(r"\b(?:where\s+(?:is|are)|(?:latitude|longitude|coordinates?)\s+(?:of|for))\b", prompt, re.I))
    def evidence_has_requested_detail(name, data):
        if name != "list_receivers" or not needs_coordinates or data.get("status") != "ok":
            return True
        rows = data.get("receivers")
        return data.get("detail") == "coordinates" or (isinstance(rows, list) and any(
            isinstance(row, dict) and row.get("coordinates") for row in rows))

    scope_matches = lambda data: _matches_explicit_scope(prompt, data, allow_partial_receivers=True)
    evidence = [(event.get("name"), event["result"]) for event in trace
                if event.get("event") == "tool_result" and isinstance(event.get("result"), dict)
                and scope_matches(event["result"])
                and evidence_has_requested_detail(event.get("name"), event["result"])
                and (event.get("name") != "query_geographic_receivers" or _matches_geographic_action(prompt, event["result"]))]
    completed = {name for name, _ in evidence}
    # History is reusable for an explicit follow-up only, not for a new station/date or refresh.
    referential = re.search(r"\b(?:that|this|those|same|it|them|previous|earlier)\b", prompt, re.I)
    fresh = re.search(r"\b(?:yesterday|today|tomorrow|again now|refresh|instead|different)\b|"
                      r"\b\d{4}[-/]\d|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}\b|\b\d{1,2}:\d{2}\b", prompt, re.I)
    if referential and not fresh:
        prior = [(part.tool_name, part.content) for message in history for part in getattr(message, "parts", ())
                 if isinstance(part, ToolReturnPart) and isinstance(part.content, dict)
                 and part.tool_name != "detect_meteors"
                 and part.content.get("status", "ok") == "ok" and scope_matches(part.content)
                 and evidence_has_requested_detail(part.tool_name, part.content)
                 and (part.tool_name != "query_geographic_receivers" or _matches_geographic_action(prompt, part.content))]
        evidence.extend(prior)
        completed.update(name for name, _ in prior)
    if "detect_meteors" in needed and not fresh and (referential or _implicit_detection_followup(prompt)):
        latest = _latest_data_result(history)
        if (latest is not None and latest.tool_name == "detect_meteors" and latest.outcome == "success"
                and isinstance(latest.content, dict) and latest.content.get("status") == "ok"
                and latest.content.get("analysis_performed") is True
                and scope_matches(latest.content)):
            completed.add("detect_meteors")
            evidence.append((latest.tool_name, latest.content))
    receiver_ids = set(_RECEIVER_ID_RE.findall(prompt.upper()))
    if len(receiver_ids) > 1:
        # Each requested receiver needs actual evidence from a relevant tool.
        # A source URL mentioning another receiver is never identity evidence.
        verified_receivers = set().union(*(_receiver_identity_ids(data) for name, data in evidence if name in needed))
        if not receiver_ids.issubset(verified_receivers):
            return True
    receiver_names = _requested_receiver_names(prompt)
    if len(receiver_names) > 1:
        identities = [_identity_name_text(data) for name, data in evidence if name in needed]
        if any(not any(_name_in_identity(receiver, identity) for identity in identities)
               for receiver in receiver_names):
            return True
    if ("query_geographic_receivers" in needed
            and re.search(r"\b(?:show|display|give|fetch|retrieve|get)\b.*\bspectrogram\b", prompt, re.I)
            and re.search(r"\b(?:how many|count|number of)\b.*\b(?:stations?|receivers?)\b", prompt, re.I)):
        actions = {event["result"].get("action") for event in trace
                   if event.get("event") == "tool_result" and event.get("name") == "query_geographic_receivers"
                   and isinstance(event.get("result"), dict) and event["result"].get("status") == "ok"
                   and _matches_explicit_scope(prompt, event["result"])
                   and event["result"].get("direction") in {"north", "south", "east", "west"}
                   and re.search(r"\b" + re.escape(str(event["result"].get("direction", ""))) + r"(?:ern)?\b", prompt, re.I)}
        count_action = "active_count" if re.search(r"\b(?:active|available)\b", prompt, re.I) else "count"
        if not {"spectrogram", count_action}.issubset(actions):
            return True
    return not needed.issubset(completed)


def check_unbacked_claims(
    prompt: str,
    model_text: str,
    trace: list[dict[str, Any]],
    history: Iterable[ModelMessage] = (),
) -> list[dict[str, Any]]:
    """Inspect model_text against completed tool results and return a list of detected grounding issues."""
    issues: list[dict[str, Any]] = []
    tool_results = [event for event in trace if event.get("event") == "tool_result" and isinstance(event.get("result"), dict)]

    # 1. Check if a tool failed or returned setup_required, yet the model claimed zero/no recordings/no meteors without qualifying as unknown.
    failed_events = [
        event for event in tool_results
        if event["result"].get("status") in {"error", "setup_required", "availability_unknown", "no_availability_information", "unknown"}
    ]
    if failed_events and _FALSE_ZERO_CLAIM_RE.search(model_text):
        issues.append({
            "code": "error_reported_as_zero",
            "tools": [e["name"] for e in failed_events],
            "message": "A tool returned an error or setup_required status; the requested data is unknown, not zero.",
        })

    # A successful HTTP response can still have missing receiver/hour rows.
    # Do not turn those unknowns into zero, while allowing an explicitly
    # identified, independently known zero in a mixed network result.
    availability_zero = re.compile(
        r"\b(?:no\s+(?:available\s+)?recordings?\b|"
        r"(?:0|zero)\s+recordings?|(?:coverage|availability)\s+(?:is|was|of)\s+(?:0(?:\.0+)?%|none|zero)|"
        r"(?:0(?:\.0+)?%|zero\s+percent)\s+(?:coverage|availability)|"
        r"station\s+was\s+(?:offline|inactive|down))(?!\w)", re.I)
    for event in tool_results:
        if event.get("name") != "check_availability" or event["result"].get("status") != "ok":
            continue
        data = event["result"]
        rows = data.get("receivers") if isinstance(data.get("receivers"), list) else []
        unknown = {str(row.get("receiver_id", "")).upper() for row in rows
                   if isinstance(row, dict) and row.get("coverage") == "unknown"}
        counts = data.get("coverage_counts") if isinstance(data.get("coverage_counts"), dict) else {}
        if not unknown and not counts.get("unknown"):
            continue
        known_zero = {str(row.get("receiver_id", "")).upper() for row in rows
                      if isinstance(row, dict) and row.get("coverage") == "none"}
        named_rows = [(str(row.get("name", "")), str(row.get("receiver_id", "")).upper())
                      for row in rows if isinstance(row, dict) and row.get("name") and row.get("receiver_id")]
        for sentence in re.split(r"(?<=[.!?])\s+|\n", model_text):
            if not availability_zero.search(sentence):
                continue
            ids = set(_RECEIVER_ID_RE.findall(sentence.upper()))
            if not ids:
                ids = {rid for name, rid in named_rows if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", sentence, re.I)}
            if not ids:
                ids = set(_RECEIVER_ID_RE.findall(prompt.upper()))
            if not ids:
                ids = {rid for name, rid in named_rows if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", prompt, re.I)}
            if ids and ids <= known_zero and not ids.intersection(unknown):
                continue
            issues.append({"code": "unknown_availability_reported_as_zero",
                           "message": "Missing availability rows are unknown, not zero."})
            break

    # A verified explicit-zero hour is known absence, not unknown coverage.
    # Restrict this correction to one receiver and a direct claim about archive
    # coverage; e.g. a separate statement that current uptime is unknown is fine.
    unknown_coverage = re.compile(
        r"\b(?:coverage|(?:(?:archive|recording)\s+)?availability)"
        r"(?:\s+(?:for|at)\s+(?:the\s+)?[A-Za-zÀ-ÿ][\wÀ-ÿ'’-]*(?:\s+\d+)?)?\s*"
        r"(?:(?:is|was|reported|shown|as|being|the|an|a)\s+|[:=]\s*|\*\*){0,6}"
        r"unknown(?:[-\s]+(?:equivalent|to[-\s]+zero))?\b|"
        r"\bunknown(?:[-\s]+(?:equivalent|to[-\s]+zero))?\s+(?:archive\s+|recording\s+)?coverage\b", re.I)
    for event in tool_results:
        data = event.get("result", {})
        rows = data.get("receivers") if isinstance(data.get("receivers"), list) else []
        if (event.get("name") != "check_availability" or data.get("status") != "ok"
                or data.get("recording_availability_checked") is not True or len(rows) != 1
                or not isinstance(rows[0], dict) or rows[0].get("coverage") != "none"
                or isinstance(rows[0].get("availability_percent"), bool)
                or rows[0].get("availability_percent") != 0):
            continue
        counts = data.get("coverage_counts")
        if isinstance(counts, dict) and counts.get("unknown"):
            continue
        receiver_id = str(rows[0].get("receiver_id", "")).upper()
        for sentence in re.split(r"(?<=[.!?])\s+|[;\n]", model_text):
            missing_rows = re.search(r"\b(?:no|zero)\s+(?:(?:usable|hourly|archive|availability)\s+){1,3}rows?\b", sentence, re.I)
            if not unknown_coverage.search(sentence) and not missing_rows:
                continue
            if missing_rows and re.search(r"\b(?:not|aren['’]t|weren['’]t)\s+(?:true\s+that\s+)?(?:there\s+(?:are|were)\s+)?no\b", sentence, re.I):
                continue
            if missing_rows and not _matches_explicit_scope(sentence, data):
                continue
            if (re.search(r"\b(?:current|live)\s+(?:station\s+)?availability\b", sentence, re.I)
                    and not re.search(r"\b(?:archive|recording)\s+(?:availability|coverage)\b", sentence, re.I)):
                continue
            ids = set(_RECEIVER_ID_RE.findall(sentence.upper()))
            if (ids and ids != {receiver_id}) or re.search(r"\b(?:another|other|different)\s+(?:station|receiver)\b", sentence, re.I):
                continue
            names = _requested_receiver_names(sentence)
            subject = re.search(r"\b([A-Z][A-Za-zÀ-ÿ'’-]+(?:\s+\d+)?)\s+(?:has|had|shows|showed|reports|reported)\b", sentence)
            if subject and subject.group(1).split()[0].casefold() not in _GENERIC_STATION_WORDS:
                names.append(subject.group(1).casefold())
            identity = _identity_name_text({"receiver_name": rows[0].get("name"), "receiver_id": receiver_id})
            if names and not any(_name_in_identity(name, identity) for name in names):
                continue
            issues.append({"code": "explicit_zero_reported_as_missing_rows" if missing_rows else "known_zero_reported_as_unknown",
                           "message": "The verified archive row has explicit zero coverage, not unknown coverage."})
            break

    # 2. Check scientific claim mismatches (sample_count, nominal_beacon_minus_lo_hz, receiver coverage, unsupported deletion).
    sci_flags = scientific_flags(model_text, tool_results, {"prompt": prompt})
    for flag in sci_flags:
        issues.append(flag)

    # 3. Check for fabricated URLs when tools ran.
    known_urls, known_receivers = known_evidence_tokens(trace, history)
    for match in _RECEIVER_ID_RE.findall(prompt):
        known_receivers.add(match)

    if tool_results:
        mentioned_urls = {u.rstrip(".,;:") for u in _URL_RE.findall(model_text)}
        invented_urls = [
            u for u in mentioned_urls
            if u not in known_urls
        ]
        if invented_urls:
            issues.append({
                "code": "invented_url",
                "urls": sorted(invented_urls),
                "message": "The model cited a URL that was not returned by any tool.",
            })

        # 4. Check for fabricated receiver IDs when catalog/availability/file tools ran.
        mentioned_receivers = set(_RECEIVER_ID_RE.findall(model_text))
        if known_receivers:
            invented_receivers = sorted(mentioned_receivers - known_receivers)
            if invented_receivers:
                issues.append({
                    "code": "invented_receiver_id",
                    "receiver_ids": invented_receivers,
                    "message": f"Receiver ID(s) {', '.join(invented_receivers)} were not present in the tool results.",
                })

    # 5. If no tool ran at all (and no prior tool results exist in session), flag if the model invents specific URLs or receiver IDs.
    if not tool_results and not any(isinstance(part, ToolReturnPart) for m in history for part in getattr(m, "parts", ())):
        mentioned_urls = {u.rstrip(".,;:") for u in _URL_RE.findall(model_text)} - known_urls
        mentioned_receivers = set(_RECEIVER_ID_RE.findall(model_text)) - known_receivers
        if mentioned_urls or mentioned_receivers:
            issues.append({
                "code": "unbacked_zero_tool_fabrication",
                "urls": sorted(mentioned_urls),
                "receiver_ids": sorted(mentioned_receivers),
                "message": "The model cited specific receiver IDs or URLs without running any tool.",
            })

    return issues


def apply_grounding_guard(
    prompt: str,
    text: str,
    trace: list[dict[str, Any]],
    history: Iterable[ModelMessage] = (),
    *,
    already_verified_events: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Sanitize fabricated URLs and append deterministic verified corrections when grounding issues are detected."""
    issues = check_unbacked_claims(prompt, text, trace, history)
    if not issues:
        return text, []

    known_urls, _ = known_evidence_tokens(trace, history)
    cleaned = text
    # Strip fabricated URLs so the user never clicks a hallucinated link.
    if any(issue["code"] in {"invented_url", "unbacked_zero_tool_fabrication"} for issue in issues):
        def _replace_url(match: re.Match[str]) -> str:
            raw = match.group(0)
            url = raw.rstrip(".,;:")
            suffix = raw[len(url):]
            if url in known_urls:
                return raw
            return f"`[unverified URL removed]`{suffix}"
        cleaned = _URL_RE.sub(_replace_url, cleaned)

    correction_notes: list[str] = []
    for issue in issues:
        code = issue["code"]
        if code == "error_reported_as_zero":
            correction_notes.append(
                "**Grounding correction:** The archive/tool lookup returned an error or setup issue, "
                "which means the result is **unknown** (not zero or unavailable)."
            )
        elif code == "receiver_coverage_mismatch":
            correction_notes.append(
                f"**Grounding correction:** `{issue['receiver_id']}` has **{issue['expected']}** coverage "
                f"in the tool result (not `{issue['claimed']}`)."
            )
        elif code == "sample_count_mismatch":
            correction_notes.append(
                f"**Grounding correction:** Verified sample count from the tool is **{', '.join(str(x) for x in issue['expected'])}**."
            )
        elif code == "nominal_offset_mismatch":
            correction_notes.append(
                f"**Grounding correction:** Verified nominal `beacon - LO` offset is **{', '.join(f'{x:g} Hz' for x in issue['expected_hz'])}**."
            )
        elif code == "invented_receiver_id":
            correction_notes.append(
                f"**Grounding correction:** Receiver ID(s) `{', '.join(issue['receiver_ids'])}` were not returned by the tool."
            )
        elif code == "unsupported_deletion_claim":
            correction_notes.append(
                "**Grounding correction:** No file or cache deletion tool is implemented; no files were deleted."
            )
        elif code == "unbacked_zero_tool_fabrication":
            correction_notes.append(
                "**Grounding correction:** No BRAMS tool was executed in this turn; specific station codes or URLs above were not verified."
            )

    # Also ensure non-already-verified tool results from this turn are surfaced when an error or mismatch occurred.
    already_ids = {id(e) for e in (already_verified_events or [])}
    unshown_events = [
        e for e in trace
        if e.get("event") == "tool_result" and id(e) not in already_ids
    ]
    extra_evidence = evidence_summary(unshown_events) if unshown_events else ""

    block_parts = []
    if correction_notes:
        block_parts.append("\n".join(dict.fromkeys(correction_notes)))
    if extra_evidence and "**Verified tool results**" not in cleaned:
        block_parts.append("**Verified tool results**\n\n" + extra_evidence)
    elif extra_evidence:
        block_parts.append(extra_evidence)

    contradiction = any(issue["code"] in {"error_reported_as_zero", "unknown_availability_reported_as_zero", "known_zero_reported_as_unknown", "explicit_zero_reported_as_missing_rows", "receiver_coverage_mismatch",
                                          "sample_count_mismatch", "nominal_offset_mismatch",
                                          "invented_receiver_id", "unsupported_deletion_claim",
                                          "unbacked_zero_tool_fabrication"} for issue in issues)
    if contradiction:
        # Do not leave the false claim visible, even above a correct footnote.
        verified = evidence_summary([
            event for event in trace
            if event.get("event") == "tool_result" and isinstance(event.get("result"), dict)
            and _matches_explicit_scope(prompt, event["result"], allow_partial_receivers=True)
        ])
        cleaned = "I couldn’t verify the original answer. " + (verified or "The requested result is unknown.")
        block_parts = []
    if block_parts:
        cleaned += "\n\n---\n" + "\n\n".join(block_parts)
    return cleaned, issues
