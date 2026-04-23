from typing import override, Optional, List, Dict, Any, Set
import time
import json
import re
import tempfile
import os
import gc
import base64
import mimetypes
from pathlib import Path

import httpx

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from openai import OpenAI

from .pdf_reader import (
    extract_pdf_urls_from_text,
    download_pdf,
    read_pdf_with_pypdf,
    analyze_elements,
    get_pdf_num_pages,
    resolve_page_span,
    extract_images_with_pymupdf,
    rank_embedded_image_paths_for_figure_artifacts,
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
PDF_FIGURE_IMAGE_MIN_AREA = 15000
PDF_FIGURE_IMAGE_MIN_SHORT_SIDE = 64
PDF_FIGURE_NEIGHBOR_PAGE_RADIUS = 1
PDF_QUOTE_CHUNK_MAX_FIGURE_IMAGES = 4
PDF_READER_SAVED_DIR = ""
PDF_FIGURE_ARTIFACT_MAX_PER_PAGE = 2
PDF_TABLE_CSV_PRECOMPUTE = "1"
PRECOMPUTE_TABLE_FIGURE_MODEL = "gpt-4o-mini"
QUOTE_EXTRACTION_MODEL = "gpt-4o-mini"
OPENAI_PDF_QUOTES_TIMEOUT = 120
OPENAI_PDF_TABLE_FIGURE_TIMEOUT = 120
PDF_TABLE_PRECOMPUTE_MAX_CALLS = 48
PDF_TABLE_FIGURE_PAGE_TEXT_CHARS = 6000
PDF_TABLE_FIGURE_RENDER_MAX_SIDE = 1200
PDF_TABLE_VISION_CREATE_ARTIFACTS = "1"
PDF_TABLE_VISION_ARTIFACT_MAX = 48
PDF_TABLE_SAVE_PAGE_PNG = "0"
PDF_QUOTES_MAX_PAGE_CHARS = 40000
PDF_QUOTES_STRATEGY = "chunked"
PDF_QUOTES_CHUNK_CHARS = 12000
PDF_QUOTES_FULL_DOC_MAX_CHARS = 1000000
PDF_QUOTE_FULL_DOC_MAX_FIGURE_IMAGES = 6
PDF_SUMMARY_MAX_CHARS = 20000

DESCRIPTION = """\
This agent can read and extract information from PDF documents. It:
- Extracts PDF URLs from user messages
- Downloads PDF files from URLs
- Extracts text content and structure from PDFs using advanced parsing
- Page range: **omit `end_page` (leave unset / null) to read the whole PDF from `start_page` through the last page.** If `end_page` is set, only that span is extracted (and optional `max_pages` may tighten it further). The document may still report a larger total page count—only the requested span is processed.
- Returns extracted information so iChatBio can answer questions about the PDF content

To use this agent, simply mention a PDF URL in your message. The agent will automatically detect it, download the PDF, and extract text for analysis (full document unless you pass `end_page`).
"""


class PDFReaderParams(BaseModel):
    pdf_url: Optional[str] = Field(
        default=None,
        description="Direct URL to a PDF file. If not provided, URLs will be extracted from the request message."
    )
    pdf_artifact: Optional[Artifact] = Field(
        default=None,
        description="Artifact containing a PDF file to read instead of a URL."
    )
    # is_specific_request: bool = Field(default=False)

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
_CHUNK_FIGURE_RE = re.compile(r"(?i)\bfigure\b")


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
                f"vision_full_page_render_pdf_page_{from_pg}"
                if from_pg == pg
                else (
                    f"vision_full_page_render_pdf_page_{from_pg}_"
                    f"neighbor_of_quote_page_{pg}"
                )
            )


def _image_files_grouped_by_page(image_files: list[str]) -> dict[int, list[str]]:
    by_page: dict[int, list[str]] = {}
    for path in image_files or []:
        m = re.search(r"page_(\d{4})_img_", Path(path).name)
        if not m:
            continue
        p = int(m.group(1))
        by_page.setdefault(p, []).append(path)
    for p in by_page:
        by_page[p] = sorted(by_page[p])
    return by_page


def _image_files_grouped_by_page_for_figures(image_files: list[str]) -> dict[int, list[str]]:
    raw = _image_files_grouped_by_page(image_files)
    min_area = PDF_FIGURE_IMAGE_MIN_AREA
    min_side = PDF_FIGURE_IMAGE_MIN_SHORT_SIDE
    out: dict[int, list[str]] = {}
    for page_num, paths in raw.items():
        out[page_num] = rank_embedded_image_paths_for_figure_artifacts(
            paths,
            min_area_px=min_area,
            min_short_side_px=min_side,
        )
    return out


def _embedded_image_1based_page_from_filename(path: str) -> Optional[int]:
    m = re.search(r"page_(\d{4})_img_", Path(path).name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _collect_ranked_figure_embedded_paths(
    images_by_page: dict[int, list[str]],
    center_pages: list[int],
    *,
    span_first: int,
    span_last: int,
) -> list[str]:
    radius = PDF_FIGURE_NEIGHBOR_PAGE_RADIUS
    min_area = PDF_FIGURE_IMAGE_MIN_AREA
    min_side = PDF_FIGURE_IMAGE_MIN_SHORT_SIDE

    page_set: Set[int] = set()
    for cp in center_pages:
        if not isinstance(cp, int):
            continue
        page_set.add(cp)
        for d in range(1, radius + 1):
            page_set.add(cp - d)
            page_set.add(cp + d)

    clipped = {p for p in page_set if span_first <= p <= span_last}
    merged: list[str] = []
    seen: set[str] = set()
    for pg in sorted(clipped):
        for path in images_by_page.get(pg, []) or []:
            if path not in seen:
                seen.add(path)
                merged.append(path)

    return rank_embedded_image_paths_for_figure_artifacts(
        merged,
        min_area_px=min_area,
        min_short_side_px=min_side,
    )


def _truncate_for_vision_prompt(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "\n\n[…truncated…]"


def _build_quote_chunk_user_content(
    base_text: str,
    chunk_body: str,
    pages_in_chunk: list[int],
    page_table_csv: dict[int, str],
    images_by_page: dict[int, list[str]],
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
                    f"--- Table CSV (vision transcription of full-page render, PDF page {p}) ---\n"
                    f"{csv_s.strip()}"
                )
                had_table_csv = True
    text_body = base_text
    if extra_csv:
        text_body = base_text + "\n\n" + "\n\n".join(extra_csv)

    max_fig = PDF_QUOTE_CHUNK_MAX_FIGURE_IMAGES

    image_parts: list[dict[str, Any]] = []
    if wants_fig:
        ranked_paths = _collect_ranked_figure_embedded_paths(
            images_by_page,
            pages_in_chunk,
            span_first=span_first,
            span_last=span_last,
        )
        n_img = 0
        for path in ranked_paths:
            if n_img >= max_fig:
                break
            pt = Path(path)
            if not pt.is_file():
                continue
            try:
                raw = pt.read_bytes()
            except OSError:
                continue
            if len(raw) > 8 * 1024 * 1024:
                continue
            mime = mimetypes.guess_type(str(pt))[0] or "image/png"
            if not mime.startswith("image/"):
                mime = "image/png"
            b64 = base64.b64encode(raw).decode("ascii")
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
            n_img += 1

    if image_parts:
        intro = (
            "The block below is PDF excerpt text (and optional table CSV from a prior vision pass). "
            "Following image(s) are embedded figures from page(s) in this excerpt; "
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
    if quote_clean in excerpt_text:
        return False
    csv_blob = "".join(page_table_csv.get(p, "") for p in pages_in_chunk)
    if had_t_csv and csv_blob and quote_clean in csv_blob:
        return False
    return True


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
                await process.log(
                    f"Using PDF artifact from parameters: local_id={params.pdf_artifact.local_id}"
                )
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

            await process.log(f"Using saved outputs directory: {saved_base_dir}")

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

                        library = "pypdf"
                        strategy = "fast"
                        include_page_breaks = False
                        infer_table_structure = True
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
                        await process.log(parse_msg, data=parse_data)

                        extracted_text_path = os.path.join(
                            temp_dir, f"extracted_text_{idx + 1}.txt"
                        )

                        start_time = time.perf_counter()

                        elements, text_length = read_pdf_with_pypdf(
                            pdf_path=downloaded_path,
                            strategy=strategy,
                            include_page_breaks=include_page_breaks,
                            infer_table_structure=infer_table_structure,
                            start_page=start_page,
                            end_page=end_page_effective,
                            max_pages=max_pages_effective,
                            text_output_path=extracted_text_path,
                        )

                        extraction_time = time.perf_counter() - start_time
                        await process.log(
                            f"PDF extraction processing time: {extraction_time:.3f} seconds",
                            data={
                                "processing_time_seconds": extraction_time,
                                "library": library
                            },
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

                        raw_text = Path(extracted_text_path).read_text(
                            encoding="utf-8", errors="replace"
                        )
                        refined_plain_text = clean_pdf_extracted_text(raw_text)
                        Path(extracted_text_path).write_text(
                            refined_plain_text, encoding="utf-8"
                        )
                        text_length = len(refined_plain_text)

                        structured_blocks = self._build_structured_blocks(elements, library)
                        for blk in structured_blocks:
                            if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                                blk["text"] = clean_pdf_extracted_text(blk["text"])

                        page_texts_for_vision = self._build_page_texts_from_structured_blocks(
                            structured_blocks
                        )

                        source_label = source.get("url") or f"artifact_{idx + 1}"
                        # a table (e.g. "Table …") are rasterized and sent to the vision LLM instead.
                        table_extraction: Dict[str, Any] = {
                            "table_count": 0,
                            "table_files": [],
                            "tables_by_page": {},
                            "output_dir": "",
                            "error": "",
                        }
                        image_extraction: Dict[str, Any] = {
                            "image_count": 0,
                            "image_files": [],
                            "images_by_page": {},
                            "output_dir": "",
                            "error": "",
                        }
                        try:
                            image_extraction = extract_images_with_pymupdf(
                                pdf_path=downloaded_path,
                                output_dir=str(saved_base_dir),
                                source_name=source_label,
                                start_page=span_first,
                                end_page=span_last,
                            )
 
                        except Exception as img_exc:
                            image_extraction["error"] = str(img_exc)
                            await process.log(
                                f"Warning: Image extraction failed for {pdf_url}: {img_exc}"
                            )

                        page_table_csv: dict[int, str] = {}
                        if PYMUPDF_AVAILABLE and downloaded_path:
                            try:
                                page_table_csv = await self._precompute_page_table_csvs(
                                    process=process,
                                    pdf_path=downloaded_path,
                                    page_texts=page_texts_for_vision,
                                    span_first=span_first,
                                    span_last=span_last,
                                    request=request,
                                    saved_base_dir=str(saved_base_dir)
                                    if saved_base_dir is not None
                                    else None,
                                    source_label=source_label,
                                )
                            except Exception as pre_exc:
                                await process.log(
                                    f"Warning: table CSV precompute failed: {pre_exc}"
                                )

                        image_files_list = list(image_extraction.get("image_files") or [])

                        quote_findings: list[dict] = []
                        quote_findings = await self._extract_quotes_from_structured_blocks(
                            process=process,
                            request=request,
                            structured_blocks=structured_blocks,
                            source_library=library,
                            source_url=pdf_url,
                            pdf_path=downloaded_path,
                            span_first=span_first,
                            span_last=span_last,
                            page_table_csv=page_table_csv,
                            image_files=image_files_list,
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
                            "table_count": table_extraction.get("table_count", 0),
                            "table_files": table_extraction.get("table_files", []),
                            "table_output_dir": table_extraction.get("output_dir", ""),
                            "table_error": table_extraction.get("error", ""),
                            "image_count": image_extraction.get("image_count", 0),
                            "image_files": image_extraction.get("image_files", []),
                            "image_output_dir": image_extraction.get("output_dir", ""),
                            "image_error": image_extraction.get("error", ""),
                            "total_pdf_pages": total_pdf_pages,
                            "extract_first_page": span_first,
                            "extract_last_page": span_last,
                        }

                        all_results.append(result)

                        artifact_description = f"Extracted text content from PDF: {pdf_url}"
                        if len(pdf_sources) > 1:
                            artifact_description += f" (PDF {idx + 1} of {len(pdf_sources)})"
                        artifact_description += f" (pages {span_first}-{span_last})"
                        artifact_description += " — cleaned (line breaks normalized)"

                        text_artifact_bytes = refined_plain_text.encode("utf-8")

                        await process.create_artifact(
                            mimetype="text/plain",
                            description=artifact_description,
                            content=text_artifact_bytes,
                            metadata={
                                "source_url": pdf_url,
                                "total_elements": stats['total_elements'],
                                "element_types": stats['element_types'],
                                "text_length": text_length,
                                "strategy": strategy,
                                "library": library,
                                "pdf_index": idx + 1,
                                "total_pdfs": len(pdf_sources),
                                "total_pdf_pages": total_pdf_pages,
                                "extract_first_page": span_first,
                                "extract_last_page": span_last
                            }
                        )

                        text_artifact_bytes = b""

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
                            images_by_page_art = _image_files_grouped_by_page_for_figures(
                                result.get("image_files") or []
                            )
                            saved_base_dir_resolved = saved_base_dir.resolve()
                            max_fig_art = PDF_FIGURE_ARTIFACT_MAX_PER_PAGE
                            art_i = 0
                            for page_num in fig_pages:
                                ranked_paths = _collect_ranked_figure_embedded_paths(
                                    images_by_page_art,
                                    [page_num],
                                    span_first=span_first,
                                    span_last=span_last,
                                )
                                for image_path in ranked_paths[:max_fig_art]:
                                    p = Path(image_path)
                                    if not p.exists() or not p.is_file():
                                        continue
                                    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                                    if not mime.startswith("image/"):
                                        mime = "image/png"
                                    art_i += 1
                                    file_pg = _embedded_image_1based_page_from_filename(
                                        str(image_path)
                                    )
                                    if file_pg is not None and file_pg != page_num:
                                        fig_desc = (
                                            f"Figure image (embedded raster PDF page {file_pg}; "
                                            f"quote attribution page {page_num}): {p.name}"
                                        )
                                    else:
                                        fig_desc = (
                                            f"Figure image (page {page_num}, backed by quote "
                                            f"finding): {p.name}"
                                        )
                                    await process.create_artifact(
                                        mimetype=mime,
                                        description=fig_desc,
                                        content=p.read_bytes(),
                                        metadata={
                                            "source_url": pdf_url,
                                            "image_path": str(p),
                                            "pdf_page": page_num,
                                            "embedded_raster_pdf_page": file_pg,
                                            "figure_from_quote_finding": True,
                                            "artifact_index": art_i,
                                            "figure_selection": (
                                                "embedded_images_neighbor_pages_ranked_by_pixel_area"
                                            ),
                                        },
                                    )
                                    try:
                                        image_path_resolved = p.resolve()
                                        image_path_resolved.relative_to(saved_base_dir_resolved)
                                        image_path_resolved.unlink(missing_ok=True)
                                    except ValueError:
                                        await process.log(
                                            f"Skipping image cleanup outside saved directory: {p}"
                                        )
                                    except Exception as cleanup_exc:
                                        await process.log(
                                            f"Warning: Failed to delete extracted image file {p}: {cleanup_exc}"
                                        )
                            if fig_pages and art_i == 0:
                                await process.log(
                                    "Quote findings marked figure-relevant but no image files "
                                    "were found on disk for those page(s).",
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
                        summary += f"  - Quote findings: {len(quote_findings)}\n"
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
                            "  - Tables: detected from text cues (e.g. “Table …”); "
                            "each candidate page is rendered to an image and CSV is produced by the vision model "
                        )
                        summary += f"  - Images extracted: {result.get('image_count', 0)}\n"
                        if result.get("image_output_dir"):
                            summary += f"  - Images saved to: {result.get('image_output_dir')}\n"
                        if result.get("image_error"):
                            summary += f"  - Image extraction warning: {result.get('image_error')}\n"
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
                element_type = element.get("type", "Unknown")
                text = element.get("text", "") or ""
                page_meta = element.get("metadata", {})
                page_number = element.get("page_number") or page_meta.get("page_number") or 1
                page_idx = max(int(page_number), 1)

                if element_type == "Text" and text.strip():
                    structured.append(
                        {
                            "type": "text",
                            "text": text.strip(),
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
                            f"Downloaded artifact {artifact.local_id} to {output_path}"
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

    async def _precompute_page_table_csvs(
        self,
        process: IChatBioAgentProcess,
        pdf_path: str,
        page_texts: dict[int, str],
        span_first: int,
        span_last: int,
        request: str,
        *,
        saved_base_dir: str | None = None,
        source_label: str = "pdf",
    ) -> dict[int, str]:
        flag = PDF_TABLE_CSV_PRECOMPUTE.strip().lower()
        if flag in ("0", "false", "no", "off"):
            return {}
        if not PYMUPDF_AVAILABLE:
            return {}

        model = PRECOMPUTE_TABLE_FIGURE_MODEL
        timeout = OPENAI_PDF_TABLE_FIGURE_TIMEOUT
        max_calls = PDF_TABLE_PRECOMPUTE_MAX_CALLS
        max_page_text = PDF_TABLE_FIGURE_PAGE_TEXT_CHARS
        render_max_side = PDF_TABLE_FIGURE_RENDER_MAX_SIDE

        cue_table, _cue_fig = find_table_figure_cue_pages(page_texts)
        table_word_pages = find_pages_with_table_word(page_texts)
        span_pages = set(range(int(span_first), int(span_last) + 1))
        extra = _extra_table_pages_for_user_request(request, page_texts, span_pages)
        candidate_pages = sorted((cue_table | table_word_pages | extra) & span_pages)

        create_art = PDF_TABLE_VISION_CREATE_ARTIFACTS.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        art_max = PDF_TABLE_VISION_ARTIFACT_MAX
        art_n = 0
        save_png = PDF_TABLE_SAVE_PAGE_PNG.strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        save_dir: Path | None = None
        if save_png and saved_base_dir:
            save_dir = Path(saved_base_dir) / _safe_name(source_label, "pdf") / "table_vision"
            save_dir.mkdir(parents=True, exist_ok=True)

        sys_pre = (
            "You see a full-page raster image of one PDF page (PyMuPDF) plus the same page's extracted plain text.\n"
            "The page was chosen because its text likely refers to a table (caption or the word Table).\n"
            "Transcribe any visible data table as CSV (comma-separated; header row when clear). "
            "If there is no table in the image, return an empty tabular_csv string.\n"
            "Do not invent cells; only transcribe what is visible.\n"
            'Return ONLY JSON: {"tabular_csv": "<string>"}'
        )

        out: dict[int, str] = {}
        calls = 0
        client = OpenAI(timeout=timeout)
        req_snip = (request or "").strip()[:500]

        for page in candidate_pages:
            if calls >= max_calls:
                break
            try:
                png_bytes = render_pdf_page_to_png_bytes(
                    pdf_path, page, max_side_px=render_max_side
                )
            except Exception as exc:
                await process.log(f"Table CSV precompute: page render failed (page {page}): {exc}")
                continue
            page_text = _truncate_for_vision_prompt(page_texts.get(page, ""), max_page_text)
            b64_png = base64.b64encode(png_bytes).decode("ascii")
            user_block = (
                f"(Context) User request (may be vague): {req_snip}\n\n"
                f"Page number (1-based): {page}\n\n"
                f"Page text:\n{page_text}\n"
            )
            calls += 1
            content = ""
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_pre},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_block},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                                },
                            ],
                        },
                    ],
                    temperature=0.0,
                )
                content = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                await process.log(f"Table CSV precompute LLM error page {page}: {exc}")
                continue

            if save_dir is not None:
                try:
                    (save_dir / f"page_{page:04d}_render.png").write_bytes(png_bytes)
                except OSError as exc:
                    await process.log(f"Table CSV precompute: failed to save PNG page {page}: {exc}")

            parsed = _parse_json_object_from_response(content)
            if not isinstance(parsed, dict):
                continue
            tab_csv = str(parsed.get("tabular_csv") or "").strip()
            if tab_csv:
                out[page] = tab_csv

        if out:
            await process.log(
                f"Table CSV precompute: stored CSV for {len(out)} page(s).",
                data={
                    "model": model,
                    "vision_calls": calls,
                    "candidate_pages": len(candidate_pages),
                    "table_vision_artifacts_created": art_n,
                },
            )
        elif calls > 0:
            await process.log(
                "Table CSV precompute: vision ran but no non-empty CSV returned.",
                data={"model": model, "vision_calls": calls},
            )
        return out

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
        image_files: list[str] | None = None,
    ) -> list[dict]:
        req = (request or "").strip()
        if not req:
            await process.log("Quote extraction skipped: empty request.")
            return []

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
            return []

        span_eff_first = (
            int(span_first) if span_first is not None else min(page_texts.keys())
        )
        span_eff_last = (
            int(span_last) if span_last is not None else max(page_texts.keys())
        )

        page_table_csv = dict(page_table_csv or {})
        images_by_page = _image_files_grouped_by_page_for_figures(
            list(image_files or [])
        )
        multimodal_assets = bool(page_table_csv or images_by_page)

        quote_findings: list[dict] = []
        chunked_quote_findings: list[dict] = []
        seen: set[tuple[str, int]] = set()
        max_chars_per_page = PDF_QUOTES_MAX_PAGE_CHARS

        usage_prompt_tokens = 0
        usage_completion_tokens = 0
        usage_total_tokens = 0
        llm_request_count = 0

        _raw_strategy = PDF_QUOTES_STRATEGY
        strategy = _raw_strategy.strip().lower()
        quote_timeout = OPENAI_PDF_QUOTES_TIMEOUT
        client = OpenAI(timeout=quote_timeout)

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
                " The excerpt may additionally include table CSV (machine-transcribed from full-page images) "
                "and/or embedded figure images for pages in range; use them together with the text to answer. "
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
            for page in sorted(page_texts.keys()):
                page_text = page_texts[page]
                if max_chars_per_page > 0:
                    page_text = page_text[:max_chars_per_page]
                if not page_text.strip():
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

            per_page_wall_seconds = time.perf_counter() - per_page_wall_start

            if strategy == "chunked":
                chunk_size = PDF_QUOTES_CHUNK_CHARS
                llm_chunks = split_page_texts_into_quote_llm_chunks(
                    page_texts, max_chars_per_page, chunk_size
                )

                t_chunked = time.perf_counter()
                for ch_i, ch in enumerate(llm_chunks):
                    chunk_body = ch.get("text") or ""
                    pages_in_chunk = ch.get("pages") or []
                    if not chunk_body.strip():
                        continue
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
                        resp_c = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_content},
                            ],
                            temperature=0.0,
                        )
                        llm_request_count += 1
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
                        if j0 != -1 and j1 != -1 and j1 >= j0:
                            try:
                                parsed_c = json.loads(content_c[j0 : j1 + 1])
                            except json.JSONDecodeError:
                                parsed_c = None
                            if isinstance(parsed_c, dict):
                                raw_q = parsed_c.get("quotes", [])
                                if isinstance(raw_q, list):
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
                                        resolved_page: int | None = None
                                        for p in pages_in_chunk:
                                            ptext = page_texts.get(p, "")
                                            if max_chars_per_page > 0:
                                                ptext = ptext[:max_chars_per_page]
                                            if quote_clean in ptext:
                                                resolved_page = p
                                                break
                                        if resolved_page is None:
                                            for p in sorted(page_texts.keys()):
                                                ptext = page_texts[p]
                                                if max_chars_per_page > 0:
                                                    ptext = ptext[:max_chars_per_page]
                                                if quote_clean in ptext:
                                                    resolved_page = p
                                                    break
                                        if resolved_page is None:
                                            if pages_in_chunk:
                                                resolved_page = pages_in_chunk[0]
                                            elif page_texts:
                                                resolved_page = min(page_texts.keys())
                                            else:
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
                    except Exception as chunk_exc:
                        await process.log(
                            f"Quote extraction (chunked strategy) failed on chunk "
                            f"{ch_i + 1}/{len(llm_chunks)}: {chunk_exc}",
                            data={"model": model_name},
                        )
                chunked_wall = time.perf_counter() - t_chunked
                await process.log(
                    "Quote extraction (chunked strategy) complete",
                    data={
                        "model": model_name,
                        "llm_requests": llm_request_count,
                        "chunk_count": len(llm_chunks),
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

            return quote_findings
        except Exception as e:
            await process.log(f"Warning: Quote extraction failed: {str(e)}", data={"model": model_name})
            return []

def create_app() -> Starlette:
    agent = PDFReaderAgent()
    app = build_agent_app(agent)
    return app
