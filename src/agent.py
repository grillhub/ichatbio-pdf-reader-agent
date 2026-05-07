from typing import Optional, List, Dict, Any, Set
try:
    from typing import override
except ImportError:
    from typing_extensions import override
import asyncio
import time
import json
import re
import tempfile
import os
import gc
import base64
import hashlib
from pathlib import Path

import httpx

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from openai import OpenAI
from fuzzysearch import find_near_matches

from .pdf_reader import (
    extract_pdf_urls_from_text,
    download_pdf,
    read_pdf_with_pymupdf4llm_json,
    analyze_elements,
    get_pdf_num_pages,
    resolve_page_span,
    PYMUPDF_AVAILABLE,
    find_table_figure_cue_pages,
    find_pages_with_table_word,
    render_pdf_page_to_png_bytes,
    _safe_name,
)

from .utils.tools import (
    clean_pdf_extracted_text,
    quote_chunk_llm_user_message_for_artifact,
    split_page_texts_into_quote_llm_chunks,
)


LOCALHOST_REPLACEMENT_HOST = os.getenv("LOCALHOST_REPLACEMENT_HOST")

# Environment-backed configuration (read once at import).
PDF_TABLE_CSV_NEIGHBOR_PAGE_RADIUS = 1
PDF_FIGURE_NEIGHBOR_PAGE_RADIUS = 1
PDF_QUOTE_CHUNK_MAX_FIGURE_IMAGES = 4
PDF_READER_SAVED_DIR = ""
PDF_FIGURE_ARTIFACT_MAX_PER_PAGE = 2
QUOTE_EXTRACTION_MODEL = "gpt-oss-120b"
# QUOTE_EXTRACTION_MODEL = "gpt-4o-mini"
OPENAI_PDF_QUOTES_TIMEOUT = 120
PDF_QUOTES_MAX_PAGE_CHARS = 40000
PDF_QUOTES_STRATEGY = "chunked"
PDF_QUOTES_CHUNK_CHARS = 12000
PDF_QUOTES_LLM_BATCH_SIZE = 5
PDF_QUOTES_MAX_SEARCH_PAGES = 20
PDF_QUOTES_QUERY_TERM_COUNT = 16
PDF_QUOTES_TOP_SCORING_CHUNKS = 5
PDF_QUOTES_TOP_SCORING_PAGES = 10
ENABLED_FUZZY_SEARCH = True
ENABLED_EXHAUSTIVE_SEARCH = True

DESCRIPTION = """\
This agent can read and extract information from PDF documents. It:
- Extracts PDF URLs from user messages
- Downloads PDF files from URLs
- Extracts text content and structure from PDFs using advanced parsing
- Returns extracted information so iChatBio can answer questions about the PDF content

To use this agent, simply mention a PDF URL in your message. The agent will automatically detect it, download the PDF, and extract text for analysis.

Entrypoint parameters (read_pdf):
- pdf_url: optional direct URL to a PDF (otherwise URLs are taken from the user message).
- pdf_artifact: optional uploaded PDF artifact instead of a URL.
"""


async def _quote_finding_loading_heartbeat(
    process: IChatBioAgentProcess,
    stop: asyncio.Event,
    progress: dict[str, Any],
    interval_s: float = 60.0,
) -> None:
    """Emit quote-finding progress every interval_s until stop is set."""
    ticks = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            ticks += 1
            elapsed = int(ticks * interval_s)
            stage = str(progress.get("stage") or "Finding information in PDF")
            page = progress.get("current_page")
            pages_done = int(progress.get("pages_done") or 0)
            pages_total = int(progress.get("pages_total") or 0)
            chunk_done = int(progress.get("chunks_done") or 0)
            chunk_total = int(progress.get("chunks_total") or 0)
            if isinstance(page, int):
                await process.log(
                    f"{stage}... still loading ({elapsed}s elapsed). "
                    f"Finding information on page {page}. "
                    f"Scanned pages: {pages_done}/{pages_total}; chunks: {chunk_done}/{chunk_total}."
                )
            else:
                await process.log(
                    f"{stage}... still loading ({elapsed}s elapsed). "
                    f"Scanned pages: {pages_done}/{pages_total}; chunks: {chunk_done}/{chunk_total}."
                )


async def _pdf_stage_loading_heartbeat(
    process: IChatBioAgentProcess,
    stop: asyncio.Event,
    stage: str,
    interval_s: float = 60.0,
) -> None:
    """Emit stage progress periodically until stop is set."""
    ticks = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            ticks += 1
            elapsed = int(ticks * interval_s)
            await process.log(f"{stage}... still loading ({elapsed}s elapsed)")


class PDFReaderParams(BaseModel):
    pdf_url: Optional[str] = Field(
        default=None,
        description="Direct URL to a PDF file. If not provided, URLs will be extracted from the request message."
    )
    pdf_artifact: Optional[Artifact] = Field(
        default=None,
        description="Artifact containing a PDF file to read instead of a URL."
    )

def _coerce_llm_quote_list_item(entry: Any) -> tuple[str, str] | None:
    if isinstance(entry, str):
        q = entry.strip()
        return (q, "") if q else None
    if isinstance(entry, dict):
        raw = entry.get("text") or entry.get("verbatim") or entry.get("quote") or entry.get("quotes")
        if not isinstance(raw, str):
            return None
        q = raw.strip()
        if not q:
            return None
        r = entry.get("reason")
        reason = r.strip() if isinstance(r, str) else ""
        return (q, reason)
    return None


def _parse_json_object_from_response(content: str) -> dict | None:
    if not content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# Words split from the user request for fuzzy retrieval (never the full sentence as one term:
# find_near_matches uses a tiny max edit distance for long strings, so long phrases score 0).
_REQUEST_RETRIEVAL_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "from",
        "with",
        "all",
        "are",
        "was",
        "were",
        "not",
        "but",
        "has",
        "have",
        "had",
        "this",
        "that",
        "any",
        "use",
        "using",
        "can",
        "will",
        "may",
        "please",
        "user",
        "upload",
        "pdf",
        "file",
        "document",
        "into",
        "over",
        "such",
        "than",
        "then",
        "them",
        "they",
        "their",
        "what",
        "when",
        "where",
        "which",
        "while",
        "your",
        "you",
        "how",
        "why",
        "extract",
        "occurrences",
        "mentioned",
        "context",
        "return",
        "list",
        "find",
        "search",
        "read",
        "give",
        "show",
        "tell",
        "describe",
        "about",
        "some",
        "each",
        "every",
    }
)


def _retrieval_tokens_from_request(
    req: str, *, seen: set[str], max_extra: int = 24
) -> list[str]:
    if not (req or "").strip():
        return []
    out: list[str] = []
    for m in re.finditer(r"\w{3,}", req, flags=re.UNICODE):
        w = m.group(0)
        lw = w.lower()
        if lw in _REQUEST_RETRIEVAL_STOPWORDS:
            continue
        if lw in seen:
            continue
        seen.add(lw)
        out.append(w)
        if len(out) >= max_extra:
            break
    return out


def _export_quote_finding(finding: dict) -> dict | None:
    qt = str(finding.get("quotes", "")).strip()
    cc = finding.get("csv_content")
    has_cc = isinstance(cc, str) and cc.strip()
    if not qt and not has_cc:
        return None
    reason = finding.get("reason")
    rs = reason.strip() if isinstance(reason, str) else ""
    out: dict[str, Any] = {"quotes": qt, "page": finding.get("page"), "reason": rs}
    if has_cc:
        out["csv_content"] = cc.strip()
        src = finding.get("csv_content_source")
        if isinstance(src, str) and src.strip():
            out["csv_content_source"] = src.strip()
    typ = finding.get("type")
    if isinstance(typ, str) and typ.strip():
        out["type"] = typ.strip()
    if finding.get("figure_relevant") is True:
        out["figure_relevant"] = True
    return out


_CHUNK_TABLE_RE = re.compile(r"(?i)\btable\b")
_CHUNK_FIGURE_RE = re.compile(r"(?i)\b(?:figure|fig(?:s)?)\.?\b")


def _chunk_mentions_table(chunk_body: str) -> bool:
    return bool(chunk_body and _CHUNK_TABLE_RE.search(chunk_body))


def _chunk_mentions_figure(chunk_body: str) -> bool:
    return bool(chunk_body and _CHUNK_FIGURE_RE.search(chunk_body))


def _extra_table_pages_for_user_request(
    request: str,
    page_texts: dict[int, str],
    span_pages: set[int],
) -> set[int]:
    r = (request or "").strip().lower()
    if not r:
        return set()
    if "table" not in r:
        return set()
    extra: set[int] = set()
    for p, raw in (page_texts or {}).items():
        try:
            pi = int(p)
        except (TypeError, ValueError):
            continue
        if pi not in span_pages:
            continue
        if not isinstance(raw, str) or not raw.strip():
            continue
        if re.search(r"(?i)\btable\s*\d", raw):
            extra.add(pi)
        if re.search(r"(?i)\bresults?\b", raw) and re.search(r"(?i)\btable\b", raw):
            extra.add(pi)
    return extra


def _resolve_table_csv_for_quote_page(
    page: int,
    page_table_csv: dict[int, str],
    span_first: int,
    span_last: int,
) -> tuple[Optional[str], Optional[int]]:
    radius = PDF_TABLE_CSV_NEIGHBOR_PAGE_RADIUS
    order: list[int] = [0]
    for i in range(1, radius + 1):
        order.extend([-i, i])
    for d in order:
        pg = page + d
        if pg < span_first or pg > span_last:
            continue
        s = page_table_csv.get(pg)
        if isinstance(s, str) and s.strip():
            return s.strip(), pg
    return None, None


def _attach_precomputed_table_csv_to_findings(
    findings: list[dict],
    page_table_csv: dict[int, str],
    span_first: int,
    span_last: int,
) -> None:
    if not page_table_csv:
        return
    for f in findings:
        if not isinstance(f, dict):
            continue
        pg = f.get("page")
        if not isinstance(pg, int):
            continue
        existing = f.get("csv_content")
        if isinstance(existing, str) and existing.strip():
            continue
        csv_s, from_pg = _resolve_table_csv_for_quote_page(
            pg, page_table_csv, span_first, span_last
        )
        if csv_s and from_pg is not None:
            f["csv_content"] = csv_s
            f["csv_content_source"] = (
                f"pymupdf4llm_table_extract_pdf_page_{from_pg}"
                if from_pg == pg
                else (
                    f"pymupdf4llm_table_extract_pdf_page_{from_pg}_"
                    f"neighbor_of_quote_page_{pg}"
                )
            )


def _embedded_images_grouped_by_page(
    embedded_images_by_page: dict[int, list[dict[str, str]]] | None,
) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    if not isinstance(embedded_images_by_page, dict):
        return out
    for k, items in embedded_images_by_page.items():
        try:
            page_num = int(k)
        except (TypeError, ValueError):
            continue
        if page_num < 1:
            continue
        clean_items: list[dict[str, str]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            b64 = str(item.get("base64") or "").strip()
            if not b64:
                continue
            mime = str(item.get("mime") or "image/png").strip()
            if not mime.startswith("image/"):
                mime = "image/png"
            clean_items.append({"base64": b64, "mime": mime})
        if clean_items:
            out[page_num] = clean_items
    return out


def _collect_figure_embedded_images(
    images_by_page: dict[int, list[dict[str, str]]],
    center_pages: list[int],
    *,
    span_first: int,
    span_last: int,
) -> list[dict[str, str]]:
    radius = PDF_FIGURE_NEIGHBOR_PAGE_RADIUS

    page_set: Set[int] = set()
    for cp in center_pages:
        if not isinstance(cp, int):
            continue
        page_set.add(cp)
        for d in range(1, radius + 1):
            page_set.add(cp - d)
            page_set.add(cp + d)

    clipped = {p for p in page_set if span_first <= p <= span_last}
    merged: list[dict[str, str]] = []
    for pg in sorted(clipped):
        for img in images_by_page.get(pg, []) or []:
            if isinstance(img, dict):
                merged.append(img)
    return merged


def _truncate_for_vision_prompt(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "\n\n[…truncated…]"


def _build_quote_chunk_user_content(
    base_text: str,
    chunk_body: str,
    pages_in_chunk: list[int],
    page_table_csv: dict[int, str],
    images_by_page: dict[int, list[dict[str, str]]],
    *,
    span_first: int,
    span_last: int,
) -> tuple[Any, bool, bool]:
    page_table_csv = page_table_csv or {}
    images_by_page = images_by_page or {}

    wants_table = _chunk_mentions_table(chunk_body)
    wants_fig = _chunk_mentions_figure(chunk_body)

    extra_csv: list[str] = []
    had_table_csv = False
    if wants_table:
        for p in pages_in_chunk:
            csv_s = page_table_csv.get(p)
            if isinstance(csv_s, str) and csv_s.strip():
                extra_csv.append(
                    f"--- Table CSV (from pymupdf4llm table.extract, PDF page {p}) ---\n"
                    f"{csv_s.strip()}"
                )
                had_table_csv = True
    text_body = base_text
    if extra_csv:
        text_body = base_text + "\n\n" + "\n\n".join(extra_csv)

    max_fig = PDF_QUOTE_CHUNK_MAX_FIGURE_IMAGES

    image_parts: list[dict[str, Any]] = []
    if wants_fig:
        embedded_images = _collect_figure_embedded_images(
            images_by_page,
            pages_in_chunk,
            span_first=span_first,
            span_last=span_last,
        )
        n_img = 0
        for img in embedded_images:
            if n_img >= max_fig:
                break
            if not isinstance(img, dict):
                continue
            b64 = str(img.get("base64") or "").strip()
            if not b64:
                continue
            mime = str(img.get("mime") or "image/png").strip()
            if not mime.startswith("image/"):
                mime = "image/png"
            if len(b64) > 11 * 1024 * 1024:
                continue
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
            n_img += 1

    if image_parts:
        intro = (
            "The block below is PDF excerpt text (and optional table CSV from parsed table blocks). "
            "Following image(s) are embedded figures from pymupdf4llm JSON on page(s) in this excerpt; "
            "figure captions such as \"Figure 1: …\" may appear only in the text. "
            "Read each image and extract information relevant to the user request.\n\n"
        )
        return (
            [{"type": "text", "text": intro + text_body}, *image_parts],
            had_table_csv,
            True,
        )

    return text_body, had_table_csv, False


def _quote_matches_excerpt(
    quote_clean: str,
    chunk_body: str,
    pages_in_chunk: list[int],
    page_texts: dict[int, str],
    max_chars_per_page: int,
    page_table_csv: dict[int, str],
    had_figure_images: bool,
    had_table_csv: bool,
) -> bool:
    if quote_clean in chunk_body:
        return True
    if had_table_csv:
        csv_blob = "".join(page_table_csv.get(p, "") for p in pages_in_chunk)
        if csv_blob.strip() and quote_clean in csv_blob:
            return True
    for p in pages_in_chunk:
        t = page_texts.get(p, "")
        if max_chars_per_page > 0:
            t = t[:max_chars_per_page]
        if quote_clean in t:
            return True
    if had_figure_images and len(quote_clean.strip()) >= 28:
        return True
    return False


def _finding_should_attach_figure_artifact(
    quote_clean: str,
    excerpt_text: str,
    had_fig: bool,
    had_t_csv: bool,
    pages_in_chunk: list[int],
    page_table_csv: dict[int, str],
) -> bool:
    if not had_fig:
        return False
    if _chunk_mentions_figure(quote_clean):
        return True
    if quote_clean in excerpt_text:
        return False
    csv_blob = "".join(page_table_csv.get(p, "") for p in pages_in_chunk)
    if had_t_csv and csv_blob and quote_clean in csv_blob:
        return False
    return True


def _normalize_for_quote_match(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def _resolve_quote_page(
    quote_clean: str,
    pages_in_chunk: list[int],
    page_texts: dict[int, str],
    max_chars_per_page: int,
) -> int | None:
    # 1) Exact substring on in-chunk pages, then globally.
    for p in pages_in_chunk:
        ptext = page_texts.get(p, "")
        if max_chars_per_page > 0:
            ptext = ptext[:max_chars_per_page]
        if quote_clean in ptext:
            return p
    for p in sorted(page_texts.keys()):
        ptext = page_texts[p]
        if max_chars_per_page > 0:
            ptext = ptext[:max_chars_per_page]
        if quote_clean in ptext:
            return p

    # 2) Normalized whitespace/case exact match.
    nq = _normalize_for_quote_match(quote_clean)
    if nq:
        for p in pages_in_chunk:
            ptext = page_texts.get(p, "")
            if max_chars_per_page > 0:
                ptext = ptext[:max_chars_per_page]
            if nq in _normalize_for_quote_match(ptext):
                return p
        for p in sorted(page_texts.keys()):
            ptext = page_texts[p]
            if max_chars_per_page > 0:
                ptext = ptext[:max_chars_per_page]
            if nq in _normalize_for_quote_match(ptext):
                return p

    # 3) LLM may return clipped passages with "..." or "…": pick page
    # with the most matching fragments.
    frags = [
        _normalize_for_quote_match(x)
        for x in re.split(r"\.\.\.|…", quote_clean)
        if _normalize_for_quote_match(x) and len(_normalize_for_quote_match(x)) >= 20
    ]
    if frags:
        candidates = pages_in_chunk if pages_in_chunk else sorted(page_texts.keys())
        best_page: int | None = None
        best_score = 0
        for p in candidates:
            ptext = page_texts.get(p, "")
            if max_chars_per_page > 0:
                ptext = ptext[:max_chars_per_page]
            np = _normalize_for_quote_match(ptext)
            score = sum(1 for f in frags if f in np)
            if score > best_score:
                best_score = score
                best_page = p
        if best_page is not None and best_score > 0:
            return best_page

    return None


class PDFReaderAgent(IChatBioAgent):

    def __init__(self):
        super().__init__()

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="PDF Reader Agent",
            description="Reads and extracts information from PDF documents. Detects PDF URLs in messages, downloads them, and extracts text content for analysis.",
            icon="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/1200px-PDF_file_icon.svg.png",
            entrypoints=[
                AgentEntrypoint(
                    id="read_pdf",
                    description=DESCRIPTION,
                    parameters=PDFReaderParams
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        if entrypoint == "read_pdf":
            await self._handle_read_pdf(context, request, params)
        else:
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def _handle_read_pdf(
        self,
        context: ResponseContext,
        request: str,
        params: Optional[PDFReaderParams]
    ):
        async with context.begin_process(summary="Reading and extracting information from PDF") as process:
            process: IChatBioAgentProcess

            await process.log(f"Params: {params}")
            await process.log(f"Request: {request}")

            pdf_sources: List[Dict] = []

            if params and isinstance(params, PDFReaderParams) and params.pdf_artifact is not None:
                pdf_sources.append(
                    {
                        "kind": "artifact",
                        "artifact": params.pdf_artifact,
                    }
                )
                # await process.log(
                #     f"Using PDF artifact from parameters: local_id={params.pdf_artifact.local_id}"
                # )
            else:
                pdf_urls: List[str] = []

                if params and isinstance(params, PDFReaderParams) and params.pdf_url:
                    pdf_urls.append(params.pdf_url)
                    await process.log(f"Downloading PDF from {params.pdf_url}")
                else:
                    extracted_urls = extract_pdf_urls_from_text(request)
                    if extracted_urls:
                        pdf_urls.extend(extracted_urls)
                        await process.log(
                            f"Extracted {len(extracted_urls)} PDF URL(s) from message: {', '.join(extracted_urls)}"
                        )

                for url in pdf_urls:
                    pdf_sources.append({"kind": "url", "url": url})

            if not pdf_sources:
                await context.reply(
                    "Error: No PDF source found. Please provide a PDF artifact or a PDF URL in your message or as a parameter."
                )
                return

            all_results = []
            temp_dir = None

            configured_saved_dir = PDF_READER_SAVED_DIR
            candidate_saved_dirs: list[Path] = []
            if configured_saved_dir:
                candidate_saved_dirs.append(Path(configured_saved_dir))
            candidate_saved_dirs.append(Path(__file__).resolve().parent / "saved")
            candidate_saved_dirs.append(Path(tempfile.gettempdir()) / "ichatbio_pdf_reader_saved")

            saved_base_dir: Path | None = None
            for candidate in candidate_saved_dirs:
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    probe = candidate / ".write_test"
                    probe.write_text("ok", encoding="utf-8")
                    probe.unlink(missing_ok=True)
                    saved_base_dir = candidate
                    break
                except Exception:
                    continue

            if saved_base_dir is None:
                await context.reply(
                    "Error: could not find a writable directory for table/image outputs. "
                    "Set PDF_READER_SAVED_DIR to a writable path."
                )
                return

            # await process.log(f"Using saved outputs directory: {saved_base_dir}")

            try:
 
                temp_dir = tempfile.mkdtemp(prefix="pdf_reader_")

                for idx, source in enumerate(pdf_sources):
                    try:
                        if source["kind"] == "artifact":
                            artifact: Artifact = source["artifact"]
                            pdf_filename = f"artifact_{artifact.local_id or idx + 1}.pdf"
                            pdf_path = os.path.join(temp_dir, pdf_filename)

                            downloaded_path, effective_url = await self._download_pdf_from_artifact(
                                artifact=artifact,
                                output_path=pdf_path,
                                process=process,
                            )
                            pdf_url = effective_url or f"artifact:{artifact.local_id}"
                            await process.log("PDF downloaded successfully from artifact.")
                        else:
                            pdf_url = source["url"]
                            pdf_filename = f"pdf_{idx + 1}.pdf"
                            pdf_path = os.path.join(temp_dir, pdf_filename)

                            downloaded_path = download_pdf(pdf_url, pdf_path)
                            await process.log("PDF downloaded successfully!")

                        library = "pymupdf4llm_json"
                        strategy = "fast"
                        start_page = 1
                        end_page = None
                        max_pages = None

                        total_pdf_pages = get_pdf_num_pages(downloaded_path)

                        if end_page is None:
                            end_page_effective = total_pdf_pages
                            max_pages_effective: int | None = None
                        else:
                            end_page_effective = min(int(end_page), total_pdf_pages)
                            max_pages_effective = max_pages
                            if max_pages_effective is None:
                                max_pages_effective = max(
                                    1, int(end_page_effective) - int(start_page) + 1
                                )

                        span_first, span_last = resolve_page_span(
                            total_pdf_pages, start_page, end_page_effective, max_pages_effective
                        )
                        parse_msg = (
                            f"Parsing PDF with {library} (pages {span_first}-{span_last} "
                            f"of {total_pdf_pages} total)"
                        )
                        parse_data: dict[str, Any] = {
                            "extract_pages_first": span_first,
                            "extract_pages_last": span_last,
                            "pdf_total_pages": total_pdf_pages,
                        }
                        if span_last < total_pdf_pages or span_first > 1:
                            parse_msg += (
                                ". Extraction is limited by start_page/end_page/max_pages; "
                                "omit end_page and max_pages to process the full document (from start_page)."
                            )
                            parse_data["full_document_hint"] = (
                                "Omit end_page and max_pages for full-PDF extraction; "
                                "set start_page=1 for page 1 through last."
                            )
                        # await process.log(parse_msg, data=parse_data)
                        await process.log(parse_msg)
                        await process.log(
                            "Starting PDF text and structure extraction. This can take a while for large files."
                        )

                        pdf_pipeline_start = time.perf_counter()
                        pymupdf4llm_start_time = time.perf_counter()

                        stop_parse_heartbeat = asyncio.Event()
                        parse_heartbeat_task = asyncio.create_task(
                            _pdf_stage_loading_heartbeat(
                                process,
                                stop_parse_heartbeat,
                                "Extracting PDF content with pymupdf4llm_json",
                            )
                        )
                        try:
                            (
                                elements,
                                text_length,
                                page_table_csv,
                                embedded_images_by_page,
                            ) = await asyncio.to_thread(
                                read_pdf_with_pymupdf4llm_json,
                                downloaded_path,
                                start_page,
                                end_page_effective,
                                max_pages_effective,
                            )
                        finally:
                            stop_parse_heartbeat.set()
                            await parse_heartbeat_task

                        extraction_time = time.perf_counter() - pymupdf4llm_start_time
                        await process.log(
                            f"PDF extraction processing time: {extraction_time:.3f} seconds",
                            # data={
                            #     "processing_time_seconds": extraction_time,
                            #     "library": library
                            # },
                        )

                        if not elements:
                            await process.log(f"Warning: No elements extracted from PDF {pdf_url}")
                            all_results.append({
                                "url": pdf_url,
                                "success": False,
                                "error": "Failed to extract elements from PDF"
                            })
                            continue
                        
                        stats = analyze_elements(elements)

                        await process.log(
                            f"Extracted {stats['total_elements']} elements and {text_length} characters of text from PDF",
                            data={
                                "total_elements": stats['total_elements'],
                                "text_length": text_length,
                            },
                        )

                        structured_blocks = self._build_structured_blocks(elements, library)
                        for blk in structured_blocks:
                            if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                                blk["text"] = clean_pdf_extracted_text(blk["text"])

                        table_count_from_elements = sum(
                            1
                            for e in elements
                            if isinstance(e, dict) and str(e.get("type", "")).lower() == "table"
                        )
                        table_extraction: Dict[str, Any] = {
                            "table_count": table_count_from_elements,
                            "table_files": [],
                            "tables_by_page": {},
                            "output_dir": "",
                            "error": "",
                        }
                        image_extraction_seconds = 0.0
                        embedded_images_by_page = _embedded_images_grouped_by_page(
                            embedded_images_by_page
                        )
                        embedded_image_count = sum(
                            len(v) for v in embedded_images_by_page.values()
                        )
                        # await process.log(
                        #     "Embedded images extracted from pymupdf4llm JSON.",
                        #     data={
                        #         "embedded_image_count": embedded_image_count,
                        #         "embedded_image_pages": len(embedded_images_by_page),
                        #     },
                        # )

                        # page_table_csv = dict(page_table_csv or {})
                        # table_csv_precompute_seconds = 0.0

                        # await process.log(
                        #     "Table CSV extracted from pymupdf4llm JSON (no vision precompute).",
                        #     data={
                        #         "tables_detected": table_count_from_elements,
                        #         "pages_with_table_csv": len(page_table_csv),
                        #         "table_csv_precompute_seconds": table_csv_precompute_seconds,
                        #     },
                        # )

                        quote_findings: list[dict] = []
                        quote_extract_start = time.perf_counter()
                        (
                            quote_findings,
                            fuzzy_search_seconds,
                            quote_extraction_stats,
                        ) = await self._extract_quotes_from_structured_blocks(
                            process=process,
                            request=request,
                            structured_blocks=structured_blocks,
                            source_library=library,
                            source_url=pdf_url,
                            pdf_path=downloaded_path,
                            span_first=span_first,
                            span_last=span_last,
                            page_table_csv=page_table_csv,
                            embedded_images_by_page=embedded_images_by_page,
                        )
                        quote_extraction_seconds = time.perf_counter() - quote_extract_start
                        total_pdf_pipeline_seconds = (
                            time.perf_counter() - pdf_pipeline_start
                        )
                        await process.log(
                            "PDF per-file timing summary",
                            data={
                                # "source_url": pdf_url,
                                "pymupdf4llm_extract_seconds": round(extraction_time, 4),
                                "image_extract_seconds": round(image_extraction_seconds, 4),
                                "table_csv_precompute_seconds": round(
                                    0, 4
                                ),
                                "quote_extraction_seconds": round(
                                    quote_extraction_seconds, 4
                                ),
                                "fuzzy_search_seconds": round(fuzzy_search_seconds, 4),
                                "total_pdf_pipeline_seconds": round(
                                    total_pdf_pipeline_seconds, 4
                                ),
                            },
                        )

                        result = {
                            "url": pdf_url,
                            "success": True,
                            "library": library,
                            "total_elements": stats['total_elements'],
                            "element_types": stats['element_types'],
                            "text_length": text_length,
                            "strategy": strategy,
                            "quote_findings": quote_findings,
                            "quote_extraction_stats": quote_extraction_stats,
                            "table_count": table_extraction.get("table_count", 0),
                            "table_files": table_extraction.get("table_files", []),
                            "table_output_dir": table_extraction.get("output_dir", ""),
                            "table_error": table_extraction.get("error", ""),
                            "image_count": embedded_image_count,
                            "image_files": [],
                            "image_output_dir": "",
                            "image_error": "",
                            "total_pdf_pages": total_pdf_pages,
                            "extract_first_page": span_first,
                            "extract_last_page": span_last,
                            "timing_seconds": {
                                "pymupdf4llm_extract": round(extraction_time, 4),
                                "image_extract": round(image_extraction_seconds, 4),
                                "table_csv_precompute": round(
                                    0, 4
                                ),
                                "quote_extraction": round(quote_extraction_seconds, 4),
                                "fuzzy_search": round(fuzzy_search_seconds, 4),
                                "total_pdf_pipeline": round(
                                    total_pdf_pipeline_seconds, 4
                                ),
                            },
                        }

                        all_results.append(result)

                        try:
                            qf_list = result.get("quote_findings") or []
                            fig_pages = sorted(
                                {
                                    int(f["page"])
                                    for f in qf_list
                                    if isinstance(f, dict)
                                    and f.get("figure_relevant") is True
                                    and isinstance(f.get("page"), int)
                                }
                            )
                            images_by_page_art = _embedded_images_grouped_by_page(
                                embedded_images_by_page
                            )
                            max_fig_art = PDF_FIGURE_ARTIFACT_MAX_PER_PAGE
                            art_i = 0
                            seen_image_hashes: set[str] = set()
                            for page_num in fig_pages:
                                candidate_images = _collect_figure_embedded_images(
                                    images_by_page_art,
                                    [page_num],
                                    span_first=span_first,
                                    span_last=span_last,
                                )
                                for embedded in candidate_images[:max_fig_art]:
                                    if not isinstance(embedded, dict):
                                        continue
                                    b64 = str(embedded.get("base64") or "").strip()
                                    if not b64:
                                        continue
                                    # Neighbor-page collection can overlap across
                                    # multiple figure-relevant pages; dedupe by
                                    # image content so identical figures are not
                                    # emitted as separate artifacts.
                                    fp = hashlib.sha256(b64.encode("ascii", "ignore")).hexdigest()
                                    if fp in seen_image_hashes:
                                        continue
                                    mime = str(embedded.get("mime") or "image/png").strip()
                                    if not mime.startswith("image/"):
                                        mime = "image/png"
                                    try:
                                        image_bytes = base64.b64decode(b64)
                                    except Exception:
                                        continue
                                    if not image_bytes:
                                        continue
                                    seen_image_hashes.add(fp)
                                    art_i += 1
                                    # fig_desc = (
                                    #     f"Figure image (page {page_num}, backed by quote "
                                    #     f"finding, source=pymupdf4llm_json_embed): #{art_i}"
                                    # )
                                    fig_desc = (f"Figure image (page {page_num}, backed by quote finding)")
                                    await process.create_artifact(
                                        mimetype=mime,
                                        description=fig_desc,
                                        content=image_bytes,
                                        metadata={
                                            "source_url": pdf_url,
                                            "pdf_page": page_num,
                                            "figure_from_quote_finding": True,
                                            "artifact_index": art_i,
                                            "figure_selection": (
                                                "embedded_images_neighbor_pages_ordered_as_json"
                                            ),
                                        },
                                    )
                            if fig_pages and art_i == 0:
                                await process.log(
                                    "Quote findings marked figure-relevant but no embedded images "
                                    "were available for those page(s).",
                                    data={"figure_pages": fig_pages},
                                )
                        except Exception as e:
                            await process.log(
                                f"Warning: Failed to create figure image artifacts for PDF {idx + 1}: {str(e)}"
                            )

                        try:
                            structured_content_bytes = json.dumps(
                                structured_blocks,
                                ensure_ascii=False,
                                indent=2,
                            ).encode("utf-8")
                            structured_description = (
                                f"Structured content blocks (each block is a page of the PDF): {pdf_url}"
                            )
                            if len(pdf_sources) > 1:
                                structured_description += f" (PDF {idx + 1} of {len(pdf_sources)})"

                            await process.create_artifact(
                                mimetype="application/json",
                                description=structured_description,
                                content=structured_content_bytes,
                                metadata={
                                    "source_url": pdf_url,
                                    "total_elements": stats["total_elements"],
                                    "element_types": stats["element_types"],
                                    "text_length": text_length,
                                    "strategy": strategy,
                                    "library": library,
                                    "pdf_index": idx + 1,
                                    "total_pdfs": len(pdf_sources),
                                    "schema": "structured_blocks_v1",
                                },
                            )
                            structured_content_bytes = b""
                        except Exception as e:
                            await process.log(
                                f"Warning: Failed to create structured blocks artifact for PDF {idx + 1}: {str(e)}"
                            )
                        finally:
                            elements = []
                            structured_blocks = []
                            gc.collect()

                    except Exception as e:
                        error_msg = str(e)
                        await process.log(f"Error processing PDF {pdf_url}: {error_msg}")
                        all_results.append({
                            "url": pdf_url,
                            "success": False,
                            "error": error_msg
                        })

                successful = sum(1 for r in all_results if r.get("success", False))
                failed = len(all_results) - successful

                summary = f"**PDF Reading Complete**\n\n"
                summary += f"**Total PDFs Processed:** {len(all_results)}\n"
                summary += f"**Successful:** {successful}\n"
                if failed > 0:
                    summary += f"**Failed:** {failed}\n"
                summary += "\n"

                for idx, result in enumerate(all_results):
                    summary += f"**PDF {idx + 1}:** {result['url']}\n"
                    if result.get("success"):
                        summary += f"  - Library used: {result.get('library', 'pypdf')}\n"
                        tp = result.get("total_pdf_pages")
                        ef = result.get("extract_first_page")
                        el = result.get("extract_last_page")
                        if tp is not None and ef is not None and el is not None:
                            summary += f"  - Pages extracted: {ef}-{el} (of {tp} total)\n"
                        summary += f"  - Elements extracted: {result.get('total_elements', 0)}\n"
                        summary += f"  - Text length: {result.get('text_length', 0):,} characters\n"
                        element_types = result.get('element_types', {})
                        if element_types:
                            types_str = ", ".join([f"{k} ({v})" for k, v in element_types.items()])
                            summary += f"  - Element types: {types_str}\n"
                        sections = result.get("sections_summary") or {}
                        title = sections.get("title")
                        abstract = sections.get("abstract")
                        methods = sections.get("methods")
                        conclusion = sections.get("conclusion")

                        if title:
                            summary += f"  - Inferred title: **{title}**\n"
                        if abstract:
                            summary += "  - Abstract (heuristic):\n"
                            summary += f"    {abstract}\n"
                        if methods:
                            summary += "  - Methods (heuristic):\n"
                            summary += f"    {methods}\n"
                        if conclusion:
                            summary += "  - Inferred conclusion:\n"
                            summary += f"    {conclusion}\n"
                        quote_findings = result.get("quote_findings") or []
                        quote_extraction_stats = result.get("quote_extraction_stats") or {}
                        summary += f"  - Quote findings: {len(quote_findings)}\n"
                        summary += (
                            "  - Chunks sent to LLM for quote extraction: "
                            f"{quote_extraction_stats.get('chunks_sent_to_llm', 0)}\n"
                        )
                        if quote_extraction_stats.get("fuzzy_search_enabled") is True:
                            page_scored = quote_extraction_stats.get("selected_pages_with_scores") or []
                            summary += (
                                "  - Fuzzy selected pages with scores:\n"
                            )
                            summary += (
                                f"```json\n{json.dumps(page_scored, ensure_ascii=False, indent=2, default=str)}\n```\n"
                            )
                            scored = quote_extraction_stats.get("selected_chunks_with_scores") or []
                            summary += (
                                "  - Fuzzy selected chunks (sent to LLM) with scores:\n"
                            )
                            summary += (
                                f"```json\n{json.dumps(scored, ensure_ascii=False, indent=2, default=str)}\n```\n"
                            )
                        quote_payload: list[Any] = []
                        for qf in quote_findings:
                            if isinstance(qf, dict):
                                ex = _export_quote_finding(qf)
                                quote_payload.append(ex if ex is not None else qf)
                            else:
                                quote_payload.append(qf)
                        summary += "  - Quote findings detail:\n"
                        summary += (
                            f"```json\n{json.dumps(quote_payload, ensure_ascii=False, indent=2, default=str)}\n```\n"
                        )
                        summary += (
                            "  - Tables: read directly from pymupdf4llm JSON "
                            "(boxclass=table, using table.extract converted to CSV)\n"
                        )
                        summary += f"  - Images extracted: {result.get('image_count', 0)}\n"
                        if result.get("image_output_dir"):
                            summary += f"  - Images saved to: {result.get('image_output_dir')}\n"
                        if result.get("image_error"):
                            summary += f"  - Image extraction warning: {result.get('image_error')}\n"
                        timing = result.get("timing_seconds") or {}
                        if timing:
                            summary += "  - Timing (seconds):\n"
                            summary += (
                                f"    - pymupdf4llm_extract: {timing.get('pymupdf4llm_extract', 0)}\n"
                            )
                            summary += (
                                f"    - image_extract: {timing.get('image_extract', 0)}\n"
                            )
                            summary += (
                                f"    - table_csv_precompute: {timing.get('table_csv_precompute', 0)}\n"
                            )
                            summary += (
                                f"    - quote_extraction: {timing.get('quote_extraction', 0)}\n"
                            )
                            summary += (
                                f"    - total_pdf_pipeline: {timing.get('total_pdf_pipeline', 0)}\n"
                            )
                        image_files_for_reply = result.get("image_files", []) or []
                        if image_files_for_reply:
                            image_details = {
                                "source_url": result.get("url"),
                                "image_count": result.get("image_count", 0),
                                "image_files": image_files_for_reply,
                            }
                            summary += "  - Image extraction details:\n"
                            summary += f"```json\n{json.dumps(image_details, ensure_ascii=False, indent=2)}\n```\n"
                    else:
                        summary += f"  - Error: {result.get('error', 'Unknown error')}\n"
                    summary += "\n"

                summary += (
                    "The extracted text content has been saved as an artifact. "
                    "You can now ask more detailed questions about the PDF content, "
                    "for example specific sections, methods, results, or figures."
                )

                await context.reply(summary)

            except Exception as e:
                await process.log(f"Unexpected error: {str(e)}")
                await context.reply(f"An error occurred while processing PDFs: {str(e)}")
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        await process.log(f"Warning: Failed to clean up temporary directory: {str(e)}")


    def _build_structured_blocks(self, elements, library: str) -> list[dict]:
        structured: list[dict] = []

        def _get_page_number_from_metadata(meta) -> int:
            if meta is None:
                return 1
            page = getattr(meta, "page_number", None)
            if page is None and isinstance(meta, dict):
                page = meta.get("page_number")
            if page is None:
                return 1
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                return 1
            
            return max(page_int, 1)

        for element in elements or []:
            if isinstance(element, dict):
                element_type = str(element.get("type", "Unknown"))
                text = element.get("text", "") or ""
                page_meta = element.get("metadata", {})
                page_number = element.get("page_number") or page_meta.get("page_number") or 1
                page_idx = max(int(page_number), 1)

                if element_type.lower() == "text" and text.strip():
                    structured.append(
                        {
                            "type": "text",
                            "text": text.strip(),
                            "page_number": page_idx,
                        }
                    )
                    continue
                if element_type.lower() == "table":
                    table_body = ""
                    table_csv = element.get("table_csv")
                    if isinstance(table_csv, str) and table_csv.strip():
                        table_body = table_csv.strip()
                    elif isinstance(text, str) and text.strip():
                        table_body = text.strip()
                    if table_body:
                        structured.append(
                            {
                                "type": "table",
                                "table_body": table_body,
                                "table_caption": [],
                                "table_footnote": [],
                                "page_number": page_idx,
                            }
                        )
                continue

            element_type = type(element).__name__
            meta = getattr(element, "metadata", None)
            page_idx = _get_page_number_from_metadata(meta)

            if "table" in element_type.lower():
                table_html = None
                if meta is not None:
                    table_html = getattr(meta, "text_as_html", None) or getattr(meta, "text", None)
                    if isinstance(meta, dict) and table_html is None:
                        table_html = meta.get("text_as_html") or meta.get("text")
                if table_html is None:
                    table_html = getattr(element, "text", None) or str(element)

                structured.append(
                    {
                        "type": "table",
                        "table_body": table_html,
                        "table_caption": [],
                        "table_footnote": [],
                        "page_number": page_idx,
                    }
                )
                continue

            if any(key in element_type.lower() for key in ["image", "figure", "picture", "photo"]):
                img_path = None
                if meta is not None:
                    img_path = getattr(meta, "image_path", None)
                    if isinstance(meta, dict) and img_path is None:
                        img_path = meta.get("image_path")

                structured.append(
                    {
                        "type": "image",
                        "img_path": img_path,
                        "image_caption": [],
                        "image_footnote": [],
                        "page_number": page_idx,
                    }
                )
                continue

            text = getattr(element, "text", None) or str(element)
            if text and text.strip():
                structured.append(
                    {
                        "type": "text",
                        "text": text.strip(),
                        "page_number": page_idx,
                    }
                )

        return structured

    async def _download_pdf_from_artifact(
        self,
        artifact: Artifact,
        output_path: str,
        process: IChatBioAgentProcess,
    ) -> tuple[str, Optional[str]]:

        if os.path.exists(output_path):
            await process.log(f"PDF already exists at {output_path}, skipping download.")
            urls = list(artifact.get_urls())
            effective_url = urls[0] if urls else None
            return output_path, effective_url

        urls = list(artifact.get_urls())
        if not urls:
            await process.log(
                f"Artifact {artifact.local_id} does not have any retrievable URLs."
            )
            raise ValueError("Artifact has no URLs to download from.")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            last_error: Optional[Exception] = None
            for url in urls:
                try:
                    await process.log(
                        f"Downloading PDF artifact {artifact.local_id} from {url}"
                    )
                    if "localhost" in url and LOCALHOST_REPLACEMENT_HOST:
                        url = url.replace("localhost", LOCALHOST_REPLACEMENT_HOST)
                    resp = await client.get(url)
                    if resp.is_success:
                        with open(output_path, "wb") as f:
                            f.write(resp.content)
                        await process.log(
                            f"Downloaded artifact {artifact.local_id}"
                        )
                        return output_path, url
                    else:
                        await process.log(
                            f"Failed to download artifact from {url}: "
                            f"{resp.status_code} {resp.reason_phrase}"
                        )
                except Exception as e:
                    last_error = e
                    await process.log(
                        f"Error downloading artifact {artifact.local_id} from {url}: {str(e)}"
                    )

        raise ValueError(
            f"Failed to download artifact {artifact.local_id}"
        ) from last_error

    def _expand_hint_match_context(self, page_text: str, start: int, end: int, margin: int = 400) -> str:
        lo = max(0, start - margin)
        hi = min(len(page_text), end + margin)
        cut = page_text.rfind("\n", lo, start)
        if cut != -1:
            lo = cut + 1
        cut_r = page_text.find("\n", end, hi)
        if cut_r != -1:
            hi = cut_r
        snippet = page_text[lo:hi].strip()
        return snippet if snippet else page_text[start:end].strip()

    def _verbatim_passages_for_hints(self, page_text: str, quote_hints: list[str]) -> list[str]:
        passages: list[str] = []
        seen: set[str] = set()
        for hint in quote_hints or []:
            raw = (hint or "").strip()
            if len(raw) < 2:
                continue
            parts = [p for p in re.split(r"\s+", raw) if p]
            if not parts:
                continue
            pattern = r"\s+".join(re.escape(p) for p in parts)
            try:
                for m in re.finditer(pattern, page_text, flags=re.IGNORECASE | re.DOTALL):
                    excerpt = self._expand_hint_match_context(page_text, m.start(), m.end())
                    if len(excerpt) < 12:
                        excerpt = page_text[m.start() : m.end()].strip()
                    if excerpt and excerpt not in seen:
                        seen.add(excerpt)
                        passages.append(excerpt)
            except re.error:
                continue
        return passages

    def _build_page_texts_from_structured_blocks(self, structured_blocks: list[dict]) -> dict[int, str]:
        page_texts: dict[int, str] = {}
        for block in structured_blocks or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            text = ""
            if btype == "text":
                text = block.get("text") or ""
            elif btype == "table":
                body = block.get("table_body") or ""
                if isinstance(body, str) and body.strip():
                    text = re.sub(r"<[^>]+>", " ", body)
                    text = re.sub(r"\s+", " ", text).strip()
            if not isinstance(text, str) or not text.strip():
                continue
            page_number = block.get("page_number") or 1
            try:
                page_idx = max(int(page_number), 1)
            except (TypeError, ValueError):
                page_idx = 1
            previous = page_texts.get(page_idx, "")
            page_texts[page_idx] = (previous + "\n" + text) if previous else text
        return page_texts

    def _term_max_l_dist(self, term: str) -> int:
        n = len((term or "").strip())
        if n <= 5:
            return 1
        if n <= 12:
            return 2
        return 3

    def _score_chunk_by_terms(self, chunk_text: str, terms: list[str]) -> tuple[float, float]:
        text = (chunk_text or "").lower()
        if not text:
            return 0.0, 0.0
        score = 0.0
        fuzzy_seconds = 0.0
        for term in terms:
            q = (term or "").strip().lower()
            if len(q) < 3:
                continue
            max_l = self._term_max_l_dist(q)
            fuzzy_start = time.perf_counter()
            try:
                matches = find_near_matches(q, text, max_l_dist=max_l)
            except Exception:
                matches = []
            fuzzy_seconds += time.perf_counter() - fuzzy_start
            if not matches:
                continue
            best_dist = min(getattr(m, "dist", max_l) for m in matches)
            term_weight = max(1.0, min(4.0, len(q) / 6.0))
            score += len(matches) * term_weight
            score += max(0.25, 1.5 - float(best_dist))
        return score, fuzzy_seconds

    async def _generate_query_terms_with_llm(
        self,
        process: IChatBioAgentProcess,
        client: OpenAI,
        model_name: str,
        request: str,
    ) -> list[str]:
        req = (request or "").strip()
        if not req:
            return []
        sys_msg = (
            "Generate search terms for fuzzy retrieval from a PDF. "
            "Return ONLY JSON like {\"terms\": [\"...\", \"...\"]}. "
            "Include exact phrases, synonyms, abbreviations, and key entities. "
            "Keep each term short (1-6 words), no duplicates."
        )
        user_msg = (
            f"User request:\n{req}\n\n"
            f"Return up to {PDF_QUOTES_QUERY_TERM_COUNT} high-value retrieval terms."
        )
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_json_object_from_response(content)
            raw_terms = parsed.get("terms", []) if isinstance(parsed, dict) else []
            out: list[str] = []
            seen: set[str] = set()
            if isinstance(raw_terms, list):
                for t in raw_terms:
                    s = str(t or "").strip()
                    k = s.lower()
                    if len(s) < 2 or k in seen:
                        continue
                    seen.add(k)
                    out.append(s)
                    if len(out) >= PDF_QUOTES_QUERY_TERM_COUNT:
                        break
            # Short phrases (<= 40 chars) can still fuzzy-match; long requests cannot (max edit ~3).
            if req and len(req) <= 40 and req.lower() not in seen:
                seen.add(req.lower())
                out.append(req)
            for tok in _retrieval_tokens_from_request(req, seen=seen, max_extra=24):
                out.append(tok)
            await process.log(f"Retrieval terms: {out}")
            return out
        except Exception as exc:
            await process.log(f"Term generation failed; using request fallback: {exc}")
            fb = _retrieval_tokens_from_request(req, seen=set(), max_extra=24)
            return fb if fb else ([req] if req else [])

    async def _extract_quotes_from_structured_blocks(
        self,
        process: IChatBioAgentProcess,
        request: str,
        structured_blocks: list[dict],
        source_library: str | None = None,
        source_url: str | None = None,
        *,
        pdf_path: str | None = None,
        span_first: int | None = None,
        span_last: int | None = None,
        page_table_csv: dict[int, str] | None = None,
        embedded_images_by_page: dict[int, list[dict[str, str]]] | None = None,
    ) -> tuple[list[dict], float, dict[str, Any]]:
        req = (request or "").strip()
        if not req:
            await process.log("Quote extraction skipped: empty request.")
            return [], 0.0, {}

        model_name = QUOTE_EXTRACTION_MODEL

        page_texts: dict[int, str] = {}
        for block in structured_blocks or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            text = ""
            if btype == "text":
                text = block.get("text") or ""
            elif btype == "table":
                body = block.get("table_body") or ""
                if isinstance(body, str) and body.strip():
                    text = re.sub(r"<[^>]+>", " ", body)
                    text = re.sub(r"\s+", " ", text).strip()
            if not isinstance(text, str) or not text.strip():
                continue
            page_number = block.get("page_number") or 1
            try:
                page_idx = max(int(page_number), 1)
            except (TypeError, ValueError):
                page_idx = 1
            previous = page_texts.get(page_idx, "")
            page_texts[page_idx] = (previous + "\n" + text) if previous else text

        if not page_texts:
            await process.log("No text pages available in structured_blocks for quote extraction.")
            return [], 0.0, {}

        span_eff_first = (
            int(span_first) if span_first is not None else min(page_texts.keys())
        )
        span_eff_last = (
            int(span_last) if span_last is not None else max(page_texts.keys())
        )

        page_table_csv = dict(page_table_csv or {})
        images_by_page = _embedded_images_grouped_by_page(embedded_images_by_page)
        multimodal_assets = bool(page_table_csv or images_by_page)

        quote_findings: list[dict] = []
        chunked_quote_findings: list[dict] = []
        seen: set[tuple[str, int]] = set()
        max_chars_per_page = PDF_QUOTES_MAX_PAGE_CHARS
        sorted_pages = sorted(page_texts.keys())
        progress: dict[str, Any] = {
            "stage": "Finding relevant information in PDF pages",
            "current_page": None,
            "pages_done": 0,
            "pages_total": len(sorted_pages),
            "chunks_done": 0,
            "chunks_total": 0,
        }
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            _quote_finding_loading_heartbeat(process, stop_heartbeat, progress)
        )

        usage_prompt_tokens = 0
        usage_completion_tokens = 0
        usage_total_tokens = 0
        llm_request_count = 0
        fuzzy_search_seconds_total = 0.0
        chunks_sent_to_llm = 0
        fuzzy_chunk_scores_used: list[dict[str, Any]] = []
        fuzzy_page_scores_used: list[dict[str, Any]] = []

        strategy = PDF_QUOTES_STRATEGY.strip().lower()
        client = OpenAI(timeout=OPENAI_PDF_QUOTES_TIMEOUT)

        system_message = (
            "You receive one user request string and document text (often with page markers). "
            "Decide if it asks for particular information, evidence, or passages (topic, question, keywords, "
            "'find / quote / what does it say'). "
            "If the request is ONLY a vague instruction to read or open the document with no target "
            "(e.g. 'read', 'open the PDF'), return {\"quotes\": []} only—do not invent filler quotes. "
            "If the request is specific, extract verbatim passages that truly satisfy it. "
            "STRUCTURE AND DISAMBIGUATION (critical): PDFs use many layouts—taxonomic catalogs, numbered sections, "
            "repeated labels like 'Identification.', 'Records.', 'Methods', tables, figure captions. "
            "When the user names an entity (species, drug, gene, product, section title, figure, table row), "
            "you must anchor quotes to THAT entity's block: read nearby lines before and after. "
            "The answer often sits immediately under the heading or name line that matches the request "
            "(same paragraph block or the next few lines), not under a different heading that merely shares the same label. "
            "Never return a passage from another organism, product, or section just because it contains a generic word "
            "(e.g. another 'Identification.' for a different species). "
            "Prefer a single contiguous quote that includes the anchor line (e.g. the scientific name / heading) "
            "plus the following description when that makes which entry you mean obvious—still exact substring from the text. "
            "If you cannot find text clearly tied to the requested entity/topic in this excerpt, return {\"quotes\": []}. "
            "Do not merge facts from unrelated entries; your reason must truthfully state how the quoted lines connect "
            "to the anchor (e.g. 'Follows the line naming Callista floridella'). "
            "Each verbatim passage must be copied EXACTLY from the provided text (contiguous substring): "
            "prefer full sentences when punctuation is normal; otherwise contiguous lines or list items "
            "(at least ~40 characters when possible). Do not paraphrase or fix spelling. "
            "For every quote, give a short \"reason\" tying the passage to the user's request and to the local structure. "
            'Return ONLY a JSON object. Preferred shape: {"quotes": [{"text": "<verbatim from page>", "reason": "..."}]}. '
            'Legacy strings are still accepted: {"quotes": ["<verbatim>", ...]}. '
            "If nothing in this excerpt matches the request with a clear structural anchor, return {\"quotes\": []}."
        )
        if multimodal_assets:
            system_message += (
                " The excerpt may additionally include table CSV (parsed from PDF table structure) "
                "and/or embedded figure images from the PDF JSON for pages in range; use them together with the text to answer. "
                "When the excerpt contains the word Figure, image(s) may follow the text—read them for facts "
                "relevant to the user request. When it contains Table, CSV lines may follow—quote exact CSV "
                "substrings if they answer the request."
            )
        user_message_prefix = (
            "User request (full string; may be a specific question/topic or a vague instruction):\n"
            f"{req}\n\n"
            "Use the visible structure of THIS excerpt (headings, taxon names, 'Identification.', captions, page breaks). "
            "Quote the passage that belongs to the entry the request asks about—not a different entry that shares a similar label. "
            "When the request names a taxon or term, include the name line in the quote if it appears adjacent in the text "
            "so the quote is self-explanatory. "
            "If this request is vague (read/open only), return {\"quotes\": []}. "
            "Bad (too short / wrong entry): "
            '{"quotes": ["method", "outcome"]}\n'
            "Good (anchored verbatim block + reason naming the anchor): "
            '{"quotes": [{"text": "Species X (Author, 1900) Fig. 1. Identification. Shell oval with …", '
            '"reason": "Species X is named on the preceding line; this Identification block describes that species as requested."}]}\n\n'
        )

        try:
            per_page_wall_start = time.perf_counter()
            for page in sorted_pages:
                progress["stage"] = "Scanning extracted page text for quote anchors"
                progress["current_page"] = page
                page_text = page_texts[page]
                if max_chars_per_page > 0:
                    page_text = page_text[:max_chars_per_page]
                if not page_text.strip():
                    progress["pages_done"] = int(progress.get("pages_done") or 0) + 1
                    continue

                base_u = (
                    user_message_prefix
                    + f"Page number: {page}\n\n"
                    + f"Page text:\n{page_text}"
                )
                user_content, had_t_csv, had_fig = _build_quote_chunk_user_content(
                    base_u,
                    page_text,
                    [page],
                    page_table_csv,
                    images_by_page,
                    span_first=span_eff_first,
                    span_last=span_eff_last,
                )

                for passage in self._verbatim_passages_for_hints(page_text, [req]):
                    key = (passage, page)
                    if key in seen:
                        continue
                    seen.add(key)
                    quote_findings.append({"quotes": passage, "page": page, "reason": ""})
                progress["pages_done"] = int(progress.get("pages_done") or 0) + 1

            per_page_wall_seconds = time.perf_counter() - per_page_wall_start

            if strategy == "chunked":
                chunk_size = PDF_QUOTES_CHUNK_CHARS
                scored_chunks: list[tuple[float, int, dict[str, Any]]] = []
                retrieval_terms: list[str] = []
                page_texts_for_chunks: dict[int, str] = page_texts

                page_score_lookup: dict[int, float] = {}
                page_order_for_chunks: list[int] | None = None

                if ENABLED_FUZZY_SEARCH:
                    retrieval_terms = await self._generate_query_terms_with_llm(
                        process, client, model_name, req
                    )
                    # Phase 1: fuzzy-score each page, sort by score (desc), then chunk in that order.
                    scored_pages: list[tuple[float, int]] = []
                    for p in sorted_pages:
                        pt = page_texts.get(p, "")
                        if max_chars_per_page > 0:
                            pt = pt[:max_chars_per_page]
                        score, fuzzy_seconds = self._score_chunk_by_terms(
                            pt, retrieval_terms
                        )
                        fuzzy_search_seconds_total += fuzzy_seconds
                        scored_pages.append((score, p))
                    scored_pages.sort(key=lambda row: (-row[0], row[1]))
                    for s, p in scored_pages:
                        page_score_lookup[p] = float(s)
                    # Always log the top pages by rank (even when all scores are 0).
                    fuzzy_page_scores_used = [
                        {"page": p, "score": round(s, 6)}
                        for s, p in scored_pages[:PDF_QUOTES_TOP_SCORING_PAGES]
                    ]
                    if ENABLED_EXHAUSTIVE_SEARCH:
                        # Exhaustive mode: keep all pages that have at least one
                        # "hit" by fuzzy scoring, ordered by score (desc).
                        chosen_pages = [
                            p
                            for s, p in scored_pages
                            if float(s) >= 1.0
                        ][:PDF_QUOTES_TOP_SCORING_PAGES]
                        if chosen_pages:
                            page_order_for_chunks = chosen_pages
                            page_texts_for_chunks = {
                                p: page_texts[p] for p in page_order_for_chunks
                            }
                        else:
                            # If no page reaches the threshold, fall back to legacy
                            # behavior: chunk the full PDF in natural page order.
                            page_texts_for_chunks = page_texts
                            page_order_for_chunks = None
                    else:
                        best_page_score = scored_pages[0][0] if scored_pages else 0.0
                        if best_page_score > 0.0:
                            chosen = scored_pages[:PDF_QUOTES_TOP_SCORING_PAGES]
                            page_order_for_chunks = [p for _s, p in chosen]
                            page_texts_for_chunks = {
                                p: page_texts[p] for p in page_order_for_chunks
                            }
                        else:
                            # No term hits: chunk the full PDF in natural page order (legacy behavior).
                            page_texts_for_chunks = page_texts
                            page_order_for_chunks = None

                llm_chunks = split_page_texts_into_quote_llm_chunks(
                    page_texts_for_chunks,
                    max_chars_per_page,
                    chunk_size,
                    page_iteration_order=page_order_for_chunks,
                )
                progress["chunks_total"] = len(llm_chunks)

                if ENABLED_FUZZY_SEARCH:
                    # Chunks are already ordered by page relevance; take the first K and
                    # attach scores as the sum of constituent page fuzzy scores.
                    chunks_for_scoring = (
                        llm_chunks
                        if ENABLED_EXHAUSTIVE_SEARCH
                        else llm_chunks[:PDF_QUOTES_TOP_SCORING_CHUNKS]
                    )
                    for ch_i, ch in enumerate(chunks_for_scoring):
                        pages_in = [
                            int(p)
                            for p in (ch.get("pages") or [])
                            if isinstance(p, int)
                        ]
                        agg = sum(page_score_lookup.get(p, 0.0) for p in pages_in)
                        scored_chunks.append((agg, ch_i, ch))
                        fuzzy_chunk_scores_used.append(
                            {
                                "chunk_index": ch_i + 1,
                                "score": round(agg, 6),
                                "pages": ch.get("pages") or [],
                            }
                        )
                else:
                    scored_chunks = [(1.0, ch_i, ch) for ch_i, ch in enumerate(llm_chunks)]

                await process.log(
                    "Chunk retrieval ranking prepared",
                    data={
                        "fuzzy_search_enabled": ENABLED_FUZZY_SEARCH,
                        "page_first_fuzzy": ENABLED_FUZZY_SEARCH,
                        "top_scoring_pages_limit": PDF_QUOTES_TOP_SCORING_PAGES,
                        "chunk_count": len(llm_chunks),
                        "chunks_selected_for_llm": len(scored_chunks),
                        "retrieval_terms_count": len(retrieval_terms),
                        "fuzzy_search_seconds": round(fuzzy_search_seconds_total, 4),
                        "max_pages_to_search": PDF_QUOTES_MAX_SEARCH_PAGES,
                        "batch_size": PDF_QUOTES_LLM_BATCH_SIZE,
                        "top_scoring_chunks_limit": PDF_QUOTES_TOP_SCORING_CHUNKS,
                        "selected_pages_with_scores": (
                            fuzzy_page_scores_used if ENABLED_FUZZY_SEARCH else []
                        ),
                        "selected_chunks_with_scores": (
                            fuzzy_chunk_scores_used if ENABLED_FUZZY_SEARCH else []
                        ),
                    },
                )

                t_chunked = time.perf_counter()
                pages_searched: set[int] = set()
                ranked_cursor = 0
                while ranked_cursor < len(scored_chunks):
                    progress["stage"] = "Finding quotes with chunked PDF search"
                    batch_rows: list[tuple[float, int, dict[str, Any]]] = []
                    while (
                        ranked_cursor < len(scored_chunks)
                        and len(batch_rows) < PDF_QUOTES_LLM_BATCH_SIZE
                    ):
                        row = scored_chunks[ranked_cursor]
                        ranked_cursor += 1
                        ch_pages = [int(p) for p in (row[2].get("pages") or []) if isinstance(p, int)]
                        if ENABLED_FUZZY_SEARCH:
                            merged = pages_searched | set(ch_pages)
                            if ch_pages and len(merged) > PDF_QUOTES_MAX_SEARCH_PAGES:
                                continue
                        batch_rows.append(row)
                    if not batch_rows:
                        break
                    for _score, _idx, ch in batch_rows:
                        for p in (ch.get("pages") or []):
                            if isinstance(p, int):
                                pages_searched.add(p)
                                progress["current_page"] = p

                    async def _query_chunk(chunk_row: tuple[float, int, dict[str, Any]]) -> dict[str, Any] | None:
                        _score, ch_i, ch = chunk_row
                        chunk_body = ch.get("text") or ""
                        pages_in_chunk = ch.get("pages") or []
                        if not chunk_body.strip():
                            return None
                        base_chunk = user_message_prefix + chunk_body
                        user_content, had_t_csv, had_fig = _build_quote_chunk_user_content(
                            base_chunk,
                            chunk_body,
                            pages_in_chunk,
                            page_table_csv,
                            images_by_page,
                            span_first=span_eff_first,
                            span_last=span_eff_last,
                        )
                        try:
                            resp_c = await asyncio.to_thread(
                                client.chat.completions.create,
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": user_content},
                                ],
                                temperature=0.0,
                            )
                        except Exception as chunk_exc:
                            await process.log(
                                f"Quote extraction (chunked strategy) failed on chunk "
                                f"{ch_i + 1}/{len(llm_chunks)}: {chunk_exc}",
                                data={"model": model_name},
                            )
                            return None
                        return {
                            "response": resp_c,
                            "chunk_body": chunk_body,
                            "pages_in_chunk": pages_in_chunk,
                            "had_t_csv": had_t_csv,
                            "had_fig": had_fig,
                        }

                    batch_results = await asyncio.gather(
                        *[_query_chunk(row) for row in batch_rows]
                    )
                    progress["chunks_done"] = int(progress.get("chunks_done") or 0) + len(batch_rows)
                    chunks_sent_to_llm += len([r for r in batch_results if r is not None])
                    llm_request_count += len([r for r in batch_results if r is not None])
                    for row in batch_results:
                        if row is None:
                            continue
                        resp_c = row["response"]
                        usage_c = getattr(resp_c, "usage", None)
                        if usage_c is not None:
                            pt = getattr(usage_c, "prompt_tokens", 0)
                            ct = getattr(usage_c, "completion_tokens", 0)
                            tt = getattr(usage_c, "total_tokens", 0)
                            usage_prompt_tokens += int(pt)
                            usage_completion_tokens += int(ct)
                            if tt is not None:
                                usage_total_tokens += int(tt)
                            else:
                                usage_total_tokens += int(pt) + int(ct)
                        content_c = resp_c.choices[0].message.content or ""
                        j0 = content_c.find("{")
                        j1 = content_c.rfind("}")
                        if j0 == -1 or j1 == -1 or j1 < j0:
                            continue
                        try:
                            parsed_c = json.loads(content_c[j0 : j1 + 1])
                        except json.JSONDecodeError:
                            parsed_c = None
                        if not isinstance(parsed_c, dict):
                            continue
                        raw_q = parsed_c.get("quotes", [])
                        if not isinstance(raw_q, list):
                            continue
                        chunk_body = row["chunk_body"]
                        pages_in_chunk = row["pages_in_chunk"]
                        had_fig = row["had_fig"]
                        had_t_csv = row["had_t_csv"]
                        for quote in raw_q:
                            coerced = _coerce_llm_quote_list_item(quote)
                            if coerced is None:
                                continue
                            quote_clean, quote_reason = coerced
                            if not quote_clean:
                                continue
                            if not _quote_matches_excerpt(
                                quote_clean,
                                chunk_body,
                                pages_in_chunk,
                                page_texts,
                                max_chars_per_page,
                                page_table_csv,
                                had_fig,
                                had_t_csv,
                            ):
                                continue
                            resolved_page = _resolve_quote_page(
                                quote_clean,
                                pages_in_chunk,
                                page_texts,
                                max_chars_per_page,
                            )
                            # Never assign an arbitrary page when mapping failed.
                            if resolved_page is None:
                                continue
                            ck = (quote_clean, resolved_page)
                            if ck in seen:
                                continue
                            seen.add(ck)
                            row_c: dict[str, Any] = {
                                "quotes": quote_clean,
                                "page": resolved_page,
                                "reason": quote_reason,
                            }
                            if _finding_should_attach_figure_artifact(
                                quote_clean,
                                chunk_body,
                                had_fig,
                                had_t_csv,
                                pages_in_chunk,
                                page_table_csv,
                            ):
                                row_c["figure_relevant"] = True
                            quote_findings.append(row_c)
                    if ENABLED_FUZZY_SEARCH and quote_findings:
                        break
                chunked_wall = time.perf_counter() - t_chunked
                await process.log(
                    "Quote extraction (chunked strategy) complete",
                    data={
                        "model": model_name,
                        "llm_requests": llm_request_count,
                        "chunk_count": len(llm_chunks),
                        "chunks_sent_to_llm": chunks_sent_to_llm,
                        "pages_searched": len(pages_searched),
                        "wall_seconds_chunked_llm": round(chunked_wall, 4),
                        "quotes_found_count": len(quote_findings),
                    },
                )

            _attach_precomputed_table_csv_to_findings(
                quote_findings,
                page_table_csv,
                span_eff_first,
                span_eff_last,
            )

            if quote_findings:
                try:
                    cleaned: list[dict] = []
                    for finding in quote_findings:
                        exp = _export_quote_finding(finding)
                        if exp:
                            cleaned.append(exp)
                    if cleaned:
                        body = json.dumps(
                            {"quote_findings": cleaned},
                            ensure_ascii=False,
                            indent=2,
                        )
                        await process.create_artifact(
                            mimetype="application/json",
                            description=(
                                f"Quote findings ({strategy} strategy) [{len(cleaned)} quotes]"
                            ),
                            content=(body + "\n").encode("utf-8"),
                            metadata={
                                "quote_findings": cleaned,
                                "quote_extraction_strategy": strategy,
                            },
                        )
                except Exception as art_exc:
                    await process.log(
                        f"Warning: Failed to create quote findings artifact ({strategy}): {art_exc}",
                        data={"model": model_name},
                    )

            if llm_request_count <= 0:
                await process.log(
                    "Quote extraction: no LLM requests recorded (no pages with text to scan).",
                    data={"model": model_name},
                )

            extraction_stats: dict[str, Any] = {
                "fuzzy_search_enabled": ENABLED_FUZZY_SEARCH,
                "chunks_sent_to_llm": chunks_sent_to_llm,
                "selected_chunk_count": len(fuzzy_chunk_scores_used)
                if ENABLED_FUZZY_SEARCH
                else chunks_sent_to_llm,
                "selected_chunks_with_scores": (
                    fuzzy_chunk_scores_used if ENABLED_FUZZY_SEARCH else []
                ),
                "selected_pages_with_scores": (
                    fuzzy_page_scores_used if ENABLED_FUZZY_SEARCH else []
                ),
            }
            return quote_findings, fuzzy_search_seconds_total, extraction_stats
        except Exception as e:
            await process.log(f"Warning: Quote extraction failed: {str(e)}", data={"model": model_name})
            return [], 0.0, {}
        finally:
            stop_heartbeat.set()
            await heartbeat_task

def create_app() -> Starlette:
    agent = PDFReaderAgent()
    app = build_agent_app(agent)
    return app
