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


# Questions that try to take over the assistant are refused before any model call is made.
_MISUSE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(ignore|disregard|forget|override) .{0,30}(instructions|prompts?|rules|guidelines)",
        r"\byou are now\b",
        r"\b(act|behave|respond) as (if you (are|were)|an?|the)\b",
        r"\bpretend (to be|you)\b",
        r"\b(developer|debug|jailbreak|dan) mode\b",
        r"\bjailbreak\b",
        r"(reveal|show|print|repeat) (me )?(your|the) .{0,15}(prompt|instructions)",
    )
]

MISUSE_REASON = "That request tries to change how this assistant works."


def screen_question(question: str) -> str | None:
    """A refusal reason if the question looks like a prompt-injection or jailbreak attempt"""
    if any(p.search(question) for p in _MISUSE_PATTERNS):
        return MISUSE_REASON
    return None
