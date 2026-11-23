"""Stop repeated identical calls within a turn before repeating external work."""

from datetime import datetime, timezone
from functools import wraps
import inspect
import json


class RepeatedToolCall(RuntimeError):
    pass


def guard_tool(function, completed: set, progress=None):
    signature = inspect.signature(function)

    @wraps(function)
    async def guarded(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        normalized = {}
        for name, value in arguments.arguments.items():
            if isinstance(value, datetime):
                value = value.astimezone(timezone.utc).isoformat()
            elif name == "search":
                value = " ".join(value.casefold().split())
                if value == "*":
                    value = ""
            normalized[name] = value
        key = function.__name__, json.dumps(normalized, sort_keys=True, default=str)
        step = await progress.begin("tool", function.__name__, dict(arguments.arguments)) if progress else None
        if key in completed:
            if progress:
                await progress.finish(step, "blocked", reason="Repeated identical tool call; external work not repeated.")
            raise RepeatedToolCall(f"Repeated {function.__name__} with the same arguments.")
        try:
            result = function(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except BaseException as error:
            if progress:
                await progress.finish(step, "cancelled" if not isinstance(error, Exception) else "error",
                                      error_type=type(error).__name__)
            raise
        if progress:
            result_status = result.get("status") if isinstance(result, dict) else None
            await progress.finish(step, "error" if result_status == "error" else "completed", result_status=result_status)
        completed.add(key)
        if isinstance(result, dict):
            receiver_id = result.get("receiver_id")
            rows = result.get("receivers", [])
            if not receiver_id and len(rows) == 1 and isinstance(rows[0], dict):
                receiver_id = rows[0].get("receiver_id")
            if receiver_id and "search" in normalized:
                alias = {**normalized, "search": receiver_id.casefold()}
                completed.add((function.__name__, json.dumps(alias, sort_keys=True, default=str)))
        return result

    return guarded
