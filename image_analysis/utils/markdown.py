import json
import re
from typing import List, Tuple, Any, Optional

FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*([\s\S]*?)\s*```",  # fenced blocks, with or without "json"
    re.MULTILINE,
)

INLINE_RE = re.compile(
    r"`\s*({[\s\S]*?}|\[[\s\S]*?\])\s*`"      # inline `{...}` or `[...]` in single backticks
)

def _looks_like_json(s: str) -> bool:
    s_strip = s.strip()
    return (s_strip.startswith("{") and s_strip.endswith("}")) or \
           (s_strip.startswith("[") and s_strip.endswith("]"))

def extract_json_strings(text: str) -> List[str]:
    """Return candidate JSON strings (raw text) from fenced and inline code."""
    candidates = []
    for m in FENCE_RE.finditer(text):
        payload = m.group(1).strip()
        if _looks_like_json(payload):
            candidates.append(payload)
    for m in INLINE_RE.finditer(text):
        payload = m.group(1).strip()
        if _looks_like_json(payload):
            candidates.append(payload)
    return candidates

def try_parse_json(s: str) -> Tuple[bool, Optional[Any], Optional[Exception]]:
    """Strict JSON parse first; if needed, attempt a couple of safe normalizations."""
    try:
        return True, json.loads(s), None
    except Exception as e1:
        # Light, safe normalizations (optional)
        fixed = s.strip()
        # 1) Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        # 2) Replace BOM/zero-width chars that sometimes sneak in
        fixed = fixed.replace("\ufeff", "")
        try:
            return True, json.loads(fixed), None
        except Exception as e2:
            return False, None, e2

def extract_json_objects(text: str) -> List[Any]:
    """Extract and parse all JSON objects/arrays found in markdown-ish text."""
    objs = []
    for s in extract_json_strings(text):
        ok, val, _err = try_parse_json(s)
        if ok:
            objs.append(val)
    return objs
