"""Defences for untrusted text (filing excerpts) that is placed in the model's context"""

import re

MAX_CHUNK_CHARS = 1500

# Lines that look like instructions aimed at an LLM are dropped from excerpts.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
        r"disregard (all |any )?(the )?(previous|prior|above)",
        r"you are now\b",
        r"new instructions?:",
        r"system prompt",
        r"reveal (your|the) (instructions|prompt)",
        r"</?\s*(source|filings|system)\s*>",
    )
]


def sanitize_excerpt(text: str) -> str:
    """Drop injection-looking lines and cap the length of a filing excerpt"""
    kept = [
        line for line in text.splitlines() if not any(p.search(line) for p in _INJECTION_PATTERNS)
    ]
    return "\n".join(kept).strip()[:MAX_CHUNK_CHARS]
