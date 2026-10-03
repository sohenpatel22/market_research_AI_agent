"""Cleaning and chunking of raw filing text before embedding"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def clean_filing_text(raw_text: str) -> str:
    """Collapse whitespace left over from HTML-to-text extraction"""
    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split cleaned filing text into overlapping chunks for embedding"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


_ITEM_RE = re.compile(r"(?im)^[ \t]*item[ \t]+(\d{1,2}[A-C]?)[ \t]*[.:\u2014-]?[ \t]*(.*)$")
MIN_SECTION_CHARS = 300


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a 10-K/10-Q into (section_label, text) using "Item N" headings"""
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [("Full text", text)]

    spans = [("Preamble", text[: matches[0].start()])]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((f"Item {m.group(1).upper()}", text[m.start() : end]))

    merged: list[tuple[str, str]] = []
    carry = ""
    for label, body in spans:
        body = carry + body
        if len(body.strip()) < MIN_SECTION_CHARS:
            carry = body + "\n"
            continue
        carry = ""
        merged.append((label, body))
    if carry.strip() and merged:
        merged[-1] = (merged[-1][0], merged[-1][1] + carry)
    return [(label, body) for label, body in merged if body.strip()]


def chunk_by_section(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """Section-aware chunking: returns (section_label, chunk_text) pairs"""
    return [
        (label, chunk)
        for label, body in split_sections(text)
        for chunk in chunk_text(body, chunk_size, chunk_overlap)
    ]
