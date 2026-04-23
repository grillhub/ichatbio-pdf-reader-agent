import os
import re
from typing import Any


def _pdf_line_is_heading_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _pdf_line_looks_like_dateline(s: str) -> bool:
    st = s.lstrip()
    if re.match(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        st,
        re.IGNORECASE,
    ):
        return True
    if re.match(r"^\d{1,2}(st|nd|rd|th)\b", st, re.IGNORECASE):
        return True
    return False


def _should_soft_join_pdf_lines(prev_line: str, next_line: str) -> bool:
    if not prev_line or not next_line:
        return False
    prev = prev_line.rstrip()
    nxt = next_line.lstrip()
    if not nxt:
        return False
    # Standalone all-caps / acronym lines (e.g. ORCHIDS, SFE ALTO) stay on their own line.
    if _pdf_line_is_heading_caps(nxt) and len(nxt) <= 52:
        return False
    first = nxt[0]
    if first.islower():
        return True
    # Continuation after colon/comma/semicolon (e.g. "Dear Customer:" then "I have…")
    if prev.endswith((",", ";", ":", "(")):
        return True
    # Wrapped sentence continued with a normal capitalized word ("… Document" / "Do not assume …")
    if (
        re.match(r"^[A-Z][a-z]", nxt)
        and not _pdf_line_looks_like_dateline(nxt)
        and not prev.endswith((".", "!", "?", "…"))
    ):
        return True
    # Wrapped column: long line without sentence end, next line is not a short heading-caps line.
    if len(prev) > 48 and not prev.endswith((".", "!", "?", "…")):
        if not (_pdf_line_is_heading_caps(nxt) and len(nxt) <= 36):
            return True
    return False


def clean_pdf_extracted_text(text: str) -> str:
    """
    Trim and reflow PDF-extracted text: remove hyphenation artifacts, join soft line breaks
    inside paragraphs, and collapse runs of spaces while keeping intentional single-line breaks.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Unicode soft hyphen often appears in PDFs
    t = t.replace("\u00ad", "")
    # Hyphenation at end of line: "exam-\nple" -> "example"
    t = re.sub(r"-\n\s*", "", t)
    lines = t.split("\n")
    out_chunks: list[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if buf.strip():
            out_chunks.append(re.sub(r"[ \t]+", " ", buf.strip()))
        buf = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_buf()
            out_chunks.append("")
            continue
        if not buf:
            buf = line
            continue
        if _should_soft_join_pdf_lines(buf, line):
            buf = buf.rstrip() + " " + line
        else:
            flush_buf()
            buf = line
    flush_buf()

    # Rejoin: single newlines between non-empty chunks, collapse 2+ empty strings to one blank line
    merged: list[str] = []
    empty_run = 0
    for ch in out_chunks:
        if ch == "":
            empty_run += 1
            if empty_run == 1 and merged and merged[-1] != "":
                merged.append("")
        else:
            empty_run = 0
            merged.append(ch)
    result = "\n".join(merged)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def split_page_texts_into_quote_llm_chunks(
    page_texts: dict[int, str],
    max_chars_per_page: int,
    max_chars_per_chunk: int,
) -> list[dict[str, Any]]:
    min_chunk = 2000
    if max_chars_per_chunk < min_chunk:
        max_chars_per_chunk = min_chunk

    header_reserve = 120
    max_slice = max(max_chars_per_chunk - header_reserve, 1500)

    def _ends_with_full_stop(text: str) -> bool:
        if not text:
            return False
        trimmed = text.rstrip()
        if not trimmed:
            return False
        trailing_closers = "\"'”’)]}"
        i = len(trimmed) - 1
        while i >= 0 and trimmed[i] in trailing_closers:
            i -= 1
        return i >= 0 and trimmed[i] == "."

    page_order = sorted(page_texts.keys())
    effective_page_texts: dict[int, str] = {}
    for p in page_order:
        pt = page_texts[p]
        if max_chars_per_page > 0:
            pt = pt[:max_chars_per_page]
        effective_page_texts[p] = pt if isinstance(pt, str) else ""

    for idx, p in enumerate(page_order[:-1]):
        current = effective_page_texts.get(p, "")
        if not current.strip() or _ends_with_full_stop(current):
            continue

        j = idx + 1
        while j < len(page_order):
            next_page = page_order[j]
            next_text = effective_page_texts.get(next_page, "")
            if not next_text.strip():
                j += 1
                continue

            stop_idx = next_text.find(".")
            if stop_idx >= 0:
                borrowed = next_text[: stop_idx + 1]
                effective_page_texts[p] = current.rstrip() + " " + borrowed.lstrip()
                effective_page_texts[next_page] = next_text[stop_idx + 1 :].lstrip()
                break

            effective_page_texts[p] = current.rstrip() + " " + next_text.strip()
            effective_page_texts[next_page] = ""
            current = effective_page_texts[p]
            j += 1

    chunks: list[dict[str, Any]] = []
    parts: list[str] = []
    pages_order: list[int] = []
    total_len = 0

    for p in page_order:
        pt = effective_page_texts.get(p, "")
        if not pt.strip():
            continue

        if len(pt) <= max_slice:
            slices = [pt]
        else:
            slices = []
            start = 0
            while start < len(pt):
                end = min(start + max_slice, len(pt))
                slices.append(pt[start:end])
                start = end

        for sl in slices:
            block = f"Page number: {p}\n\nPage text:\n{sl}\n\n"
            blen = len(block)
            if not parts:
                parts.append(block)
                pages_order.append(p)
                total_len = blen
                continue
            if total_len + blen > max_chars_per_chunk:
                chunks.append({"pages": list(pages_order), "text": "".join(parts)})
                parts = [block]
                pages_order = [p]
                total_len = blen
            else:
                parts.append(block)
                pages_order.append(p)
                total_len += blen
    if parts:
        chunks.append({"pages": list(pages_order), "text": "".join(parts)})
    return chunks


def quote_chunk_llm_user_message_for_artifact(
    user_message_prefix: str,
    chunk_body: str,
) -> str:
    max_chars = int(os.getenv("PDF_QUOTES_CHUNK_ARTIFACT_MAX_CHARS", "800000"))
    full_user = user_message_prefix + chunk_body
    if len(full_user) <= max_chars:
        return full_user
    return (
        full_user[:max_chars]
        + f"\n\n[... truncated for artifact; cap PDF_QUOTES_CHUNK_ARTIFACT_MAX_CHARS={max_chars} ...]\n"
    )
