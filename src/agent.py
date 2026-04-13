from typing import override, Optional, List, Dict
import time
import json
import re
import tempfile
import os
import gc
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
)
from .pdf_reader_unstructured import (
    read_pdf_with_unstructured as read_pdf_with_unstructured_lib,
    extract_text_from_elements as extract_text_from_elements_unstructured,
    analyze_elements as analyze_elements_unstructured
)

LOCALHOST_REPLACEMENT_HOST = os.getenv("LOCALHOST_REPLACEMENT_HOST")

DESCRIPTION = """\
This agent can read and extract information from PDF documents. It:
- Extracts PDF URLs from user messages
- Downloads PDF files from URLs
- Extracts text content and structure from PDFs using advanced parsing
- Supports optional page ranges (start_page, end_page, max_pages) for large PDFs; use pypdf for best efficiency on huge files
- Returns extracted information so iChatBio can answer questions about the PDF content

To use this agent, simply mention a PDF URL in your message. The agent will automatically detect it, download the PDF, and extract all text content for analysis.
"""


class PDFReaderParams(BaseModel):
    """Parameters for PDF reading operation"""
    pdf_url: Optional[str] = Field(
        default=None,
        description="Direct URL to a PDF file. If not provided, URLs will be extracted from the request message."
    )
    library: str = Field(
        default="pypdf",
        description="PDF reading library to use: 'pypdf' (default) or 'unstructured'. If not specified, defaults to 'pypdf'."
    )
    strategy: str = Field(
        default="fast",
        description="PDF parsing strategy. For unstructured: 'auto', 'hi_res', 'ocr_only', 'fast'. For pypdf: not used."
    )
    include_page_breaks: bool = Field(
        default=False,
        description="Whether to include page breaks in the extracted text"
    )
    infer_table_structure: bool = Field(
        default=True,
        description="Whether to infer and preserve table structure in the PDF"
    )
    pdf_artifact: Optional[Artifact] = Field(
        default=None,
        description="Artifact containing a PDF file to read instead of a URL."
    )
    start_page: int = Field(
        default=1,
        ge=1,
        description="1-based index of the first page to extract (default: 1).",
    )
    end_page: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "1-based inclusive last page. Omit to read through the last page of the PDF "
            "(pages start_page–N where N is the document page count), unless max_pages is set."
        ),
    )
    max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of pages to extract starting at start_page. Useful as “first N pages” when combined with start_page=1.",
    )
    quotes: list[str] = Field(
        default_factory=list,
        description=(
            "Specific words/phrases/quote hints to find in the PDF. "
            "If provided, the agent asks the LLM to extract matching verbatim quotes from the document and "
            "creates a JSON artifact with quote text and page number."
        ),
    )


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
        """Handle PDF reading requests"""
        async with context.begin_process(summary="Reading and extracting information from PDF") as process:
            process: IChatBioAgentProcess

            await process.log(f"Params: {params}")

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

                        library = params.library if params and isinstance(params, PDFReaderParams) else "pypdf"
                        strategy = params.strategy if params and isinstance(params, PDFReaderParams) else "fast"
                        include_page_breaks = params.include_page_breaks if params and isinstance(params, PDFReaderParams) else False
                        infer_table_structure = params.infer_table_structure if params and isinstance(params, PDFReaderParams) else True
                        start_page = params.start_page if params and isinstance(params, PDFReaderParams) else 1
                        end_page = params.end_page if params and isinstance(params, PDFReaderParams) else None
                        max_pages = params.max_pages if params and isinstance(params, PDFReaderParams) else None

                        library = library.lower().strip()
                        if library not in ["pypdf", "unstructured"]:
                            await process.log(f"Warning: Unknown library '{library}', defaulting to 'pypdf'")
                            library = "pypdf"

                        total_pdf_pages = get_pdf_num_pages(downloaded_path)
                        end_page_effective = end_page
                        if end_page_effective is None and max_pages is None:
                            end_page_effective = total_pdf_pages

                        max_pages_effective = max_pages
                        if max_pages_effective is None:
                            max_pages_effective = max(1, int(end_page_effective) - int(start_page) + 1)

                        span_first, span_last = resolve_page_span(
                            total_pdf_pages, start_page, end_page_effective, max_pages_effective
                        )
                        await process.log(f"Parsing PDF with {library} (pages {span_first}-{span_last} of {total_pdf_pages})")

                        extracted_text_path = os.path.join(
                            temp_dir, f"extracted_text_{idx + 1}.txt"
                        )

                        start_time = time.perf_counter()

                        if library == "unstructured":
                            elements, text_length = read_pdf_with_unstructured_lib(
                                pdf_path=downloaded_path,
                                strategy=strategy,
                                include_page_breaks=include_page_breaks,
                                infer_table_structure=infer_table_structure,
                                start_page=start_page,
                                end_page=end_page_effective,
                                max_pages=max_pages_effective,
                                text_output_path=extracted_text_path,
                            )
                        else:
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

                        if library == "unstructured":
                            stats = analyze_elements_unstructured(elements)
                        else:
                            stats = analyze_elements(elements)

                        await process.log(
                            f"Extracted {stats['total_elements']} elements and {text_length} characters of text from PDF",
                            data={
                                "total_elements": stats['total_elements'],
                                "text_length": text_length,
                            },
                        )

                        structured_blocks = self._build_structured_blocks(elements, library)

                        quote_findings: list[dict] = []
                        if params and isinstance(params, PDFReaderParams) and params.quotes:
                            quote_findings = await self._extract_quotes_from_structured_blocks(
                                process=process,
                                quote_hints=params.quotes,
                                structured_blocks=structured_blocks,
                            )

                        result = {
                            "url": pdf_url,
                            "success": True,
                            "library": library,
                            "total_elements": stats['total_elements'],
                            "element_types": stats['element_types'],
                            "text_length": text_length,
                            "strategy": strategy,
                            # "sections_summary": sections_summary,
                            "quote_findings": quote_findings,
                            "total_pdf_pages": total_pdf_pages,
                            "extract_first_page": span_first,
                            "extract_last_page": span_last,
                        }

                        all_results.append(result)

                        artifact_description = f"Extracted text content from PDF: {pdf_url}"
                        if len(pdf_sources) > 1:
                            artifact_description += f" (PDF {idx + 1} of {len(pdf_sources)})"
                        artifact_description += f" (pages {span_first}-{span_last})"

                        text_artifact_bytes = Path(extracted_text_path).read_bytes()

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
                                "extract_last_page": span_last,
                            }
                        )

                        text_artifact_bytes = b""

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
                            # await process.log(
                            #     f"Created artifact with structured content blocks from PDF {idx + 1}"
                            # )
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
                        if quote_findings:
                            summary += f"  - Quote findings: {len(quote_findings)}\n"
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
                # Clean up temporary directory
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        # await process.log(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        await process.log(f"Warning: Failed to clean up temporary directory: {str(e)}")


    def _build_structured_blocks(self, elements, library: str) -> list[dict]:
        structured: list[dict] = []

        def _get_page_number_from_metadata(meta) -> int:
            if meta is None:
                return 1
            # Unstructured metadata objects often expose attributes; fall back to dict-style access
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

            # Image or figure-like elements
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

            # Default: treat as text-like element
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
            # Try to expose at least one URL, if available
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
        """Expand a [start:end) match span to a readable verbatim excerpt (substring of page_text)."""
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

    async def _extract_quotes_from_structured_blocks(
        self,
        process: IChatBioAgentProcess,
        quote_hints: list[str],
        structured_blocks: list[dict],
    ) -> list[dict]:
        model_name = os.getenv("OPENAI_PDF_QUOTES_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        await process.log(
            f"Extracting quotes from structured document pages using model {model_name}"
        )

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

        client = OpenAI()

        quote_findings: list[dict] = []
        seen: set[tuple[str, int]] = set()
        max_chars_per_page = int(os.getenv("PDF_QUOTES_MAX_PAGE_CHARS", "40000"))
        hints_norm = {h.strip().lower() for h in (quote_hints or []) if h and h.strip()}

        usage_prompt_tokens = 0
        usage_completion_tokens = 0
        usage_total_tokens = 0
        llm_request_count = 0

        compare_full_doc = True

        system_message = (
            "Extract quotes from this document. "
            "The user may supply hints that are keywords, short phrases, or questions. "
            "Those hints might not appear word-for-word on the page; still return one or more passages "
            "that are clearly relevant (same topic, answer the implied question, or support what the hints ask about). "
            "Each returned string must be copied EXACTLY from the page text (contiguous substring): "
            "prefer a full sentence ending in . ! or ? when the page has normal punctuation; "
            "if the page uses headings, bullets, or technical lines without a period, return one or more "
            "contiguous verbatim lines or list items (at least ~40 characters when possible) instead. "
            "Do not paraphrase, summarize, or fix spelling. "
            'Return ONLY a JSON object: {"quotes": ["..."]}. '
            "If nothing on this page is relevant, return {\"quotes\": []}."
        )
        user_message_prefix = (
            f"Hints (topics or queries; they need not literally appear on the page):\n"
            f"{json.dumps(quote_hints, ensure_ascii=False)}\n\n"
            "Return full sentence(s) from the page text that best address these hints. "
            "Bad (too short / not full sentences): "
            '{"quotes": ["method", "outcome"]}\n'
            "Good (verbatim full sentence(s) from the page): "
            '{"quotes": ["We compared treatment A and treatment B using a randomized controlled design."]}\n\n'
        )

        try:
            per_page_wall_start = time.perf_counter()
            for page in sorted(page_texts.keys()):
                page_text = page_texts[page]
                if max_chars_per_page > 0:
                    page_text = page_text[:max_chars_per_page]
                if not page_text.strip():
                    continue

                user_message = (
                    user_message_prefix
                    + f"Page number: {page}\n\n"
                    + f"Page text:\n{page_text}"
                )

                # await process.log(f"System message: {system_message}")
                # await process.log(f"User message: {user_message}")

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.0,
                )
                llm_request_count += 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", 0)
                    ct = getattr(usage, "completion_tokens", 0)
                    tt = getattr(usage, "total_tokens", 0)
                    usage_prompt_tokens += int(pt)
                    usage_completion_tokens += int(ct)
                    if tt is not None:
                        usage_total_tokens += int(tt)
                    else:
                        usage_total_tokens += int(pt) + int(ct)

                content = response.choices[0].message.content or ""

                json_start = content.find("{")
                json_end = content.rfind("}")
                if json_start == -1 or json_end == -1 or json_end < json_start:
                    await process.log(
                        f"Quote extraction: no JSON object in LLM response for page {page}.",
                        data={"page": page, "preview": content[:500]},
                    )
                else:
                    try:
                        parsed = json.loads(content[json_start : json_end + 1])
                    except json.JSONDecodeError as exc:
                        await process.log(
                            f"Quote extraction: JSON parse failed on page {page}: {exc}",
                            data={"page": page, "preview": content[:500]},
                        )
                        parsed = None
                    if parsed is not None:
                        candidate_quotes = parsed.get("quotes", [])
                        if not isinstance(candidate_quotes, list):
                            await process.log(
                                f'Quote extraction: expected "quotes" list on page {page}.',
                                data={"page": page},
                            )
                        else:
                            rejected_not_substring = 0
                            for quote in candidate_quotes:
                                if not isinstance(quote, str):
                                    continue
                                quote_clean = quote.strip()
                                if not quote_clean:
                                    continue
                                if quote_clean not in page_text:
                                    rejected_not_substring += 1
                                    continue
                                if (
                                    quote_clean.lower() in hints_norm
                                    and not re.search(r"[.!?]", quote_clean)
                                ):
                                    continue

                                key = (quote_clean, page)
                                if key in seen:
                                    continue
                                seen.add(key)
                                quote_findings.append({"quotes": quote_clean, "page": page})

                for passage in self._verbatim_passages_for_hints(page_text, quote_hints):
                    key = (passage, page)
                    if key in seen:
                        continue
                    seen.add(key)
                    quote_findings.append({"quotes": passage, "page": page})

            per_page_wall_seconds = time.perf_counter() - per_page_wall_start

            if compare_full_doc:
                full_doc_max_chars = 1000000
                full_prompt_tokens = 0
                full_completion_tokens = 0
                full_total_tokens = 0
                full_doc_wall_seconds: float | None = None
                single_request_quote_findings: list[dict] = []

                page_blocks: list[str] = []
                for p in sorted(page_texts.keys()):
                    pt = page_texts[p]
                    if max_chars_per_page > 0:
                        pt = pt[:max_chars_per_page]
                    if not pt.strip():
                        continue
                    page_blocks.append(f"Page number: {p}\n\nPage text:\n{pt}")
                user_message_full = user_message_prefix + "\n\n".join(page_blocks)
                if full_doc_max_chars > 0 and len(user_message_full) > full_doc_max_chars:
                    user_message_full = user_message_full[:full_doc_max_chars]
                    await process.log(
                        "Quote extraction benchmark: truncated combined user message "
                        f"to {full_doc_max_chars} chars (PDF_QUOTES_FULL_DOC_MAX_CHARS).",
                        data={"max_chars": full_doc_max_chars},
                    )

                # await process.log(f"Full System message: {system_message}")
                # await process.log(f"Full User message: {user_message_full}")

                if user_message_full.strip() and page_blocks:
                    t_full_start = time.perf_counter()
                    try:
                        response_full = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message_full},
                            ],
                            temperature=0.0,
                        )
                        full_doc_wall_seconds = time.perf_counter() - t_full_start
                        usage_f = getattr(response_full, "usage", None)
                        if usage_f is not None:
                            pt = getattr(usage_f, "prompt_tokens", 0)
                            ct = getattr(usage_f, "completion_tokens", 0)
                            tt = getattr(usage_f, "total_tokens", 0)
                            full_prompt_tokens += int(pt)
                            full_completion_tokens += int(ct)
                            if tt is not None:
                                full_total_tokens += int(tt)
                            else:
                                full_total_tokens += int(pt) + int(ct)

                        content_f = response_full.choices[0].message.content or ""
                        js = content_f.find("{")
                        je = content_f.rfind("}")
                        if js != -1 and je != -1 and je >= js:
                            try:
                                parsed_f = json.loads(content_f[js : je + 1])
                            except json.JSONDecodeError:
                                parsed_f = None
                            if isinstance(parsed_f, dict):
                                raw_quotes = parsed_f.get("quotes", [])
                                if isinstance(raw_quotes, list):
                                    seen_single: set[tuple[str, int]] = set()
                                    for quote in raw_quotes:
                                        if not isinstance(quote, str):
                                            continue
                                        quote_clean = quote.strip()
                                        if not quote_clean:
                                            continue
                                        for p in sorted(page_texts.keys()):
                                            ptext = page_texts[p]
                                            if max_chars_per_page > 0:
                                                ptext = ptext[:max_chars_per_page]
                                            if quote_clean not in ptext:
                                                continue
                                            if (
                                                quote_clean.lower() in hints_norm
                                                and not re.search(r"[.!?]", quote_clean)
                                            ):
                                                break
                                            key = (quote_clean, p)
                                            if key in seen_single:
                                                break
                                            seen_single.add(key)
                                            single_request_quote_findings.append(
                                                {"quotes": quote_clean, "page": p}
                                            )
                                            break
                    except Exception as bench_exc:
                        await process.log(
                            f"Quote extraction benchmark (full document) failed: {bench_exc}",
                            data={"model": model_name},
                        )

                await process.log(
                    "Quote extraction: each page vs one request for all pages",
                    data={
                        "pages_scanned": len(page_texts),
                        "model": model_name,
                        "per_page_requests": {
                            "time_usage_seconds": round(per_page_wall_seconds, 4),
                            "quotes_found_count": len(quote_findings),
                            "prompt_tokens_total": usage_prompt_tokens,
                            "completion_tokens_total": usage_completion_tokens,
                            "tokens_total": usage_total_tokens,
                        },
                        "single_request_all_pages": {
                            "time_usage_seconds": (
                                round(full_doc_wall_seconds, 4)
                                if full_doc_wall_seconds is not None
                                else 0.0
                            ),
                            "quotes_found_count": len(single_request_quote_findings),
                            "prompt_tokens_total": full_prompt_tokens,
                            "completion_tokens_total": full_completion_tokens,
                            "tokens_total": full_total_tokens,
                        },
                    },
                )

                if quote_findings:
                    await process.log(
                        "Quote extraction findings (per-page).",
                        data={
                            "model": model_name,
                            "pages_scanned": len(page_texts),
                            "quote_count": len(quote_findings),
                            "quote_findings": quote_findings,
                        },
                    )
                    try:
                        cleaned_findings: list[dict] = []
                        for finding in quote_findings:
                            quote_text = str(finding.get("quotes", "")).strip()
                            quote_page = finding.get("page", None)
                            if not quote_text:
                                continue
                            cleaned_findings.append({"quotes": quote_text, "page": quote_page})

                        if cleaned_findings:
                            content = json.dumps(
                                {"quote_findings": cleaned_findings},
                                ensure_ascii=False,
                                indent=2,
                            )
                            await process.create_artifact(
                                mimetype="application/json",
                                description=f"Quote findings (per-page) [{len(cleaned_findings)} quotes]",
                                content=(content + "\n").encode("utf-8"),
                                metadata={
                                    "quote_findings": cleaned_findings,
                                },
                            )
                    except Exception as art_exc:
                        await process.log(
                            f"Warning: Failed to create aggregated per-page quote findings artifact: {art_exc}",
                            data={"model": model_name},
                        )

                if single_request_quote_findings:
                    await process.log(
                        "Quote extraction findings (single request over all pages).",
                        data={
                            "model": model_name,
                            "pages_scanned": len(page_texts),
                            "quote_count": len(single_request_quote_findings),
                            "single_request_quote_findings": single_request_quote_findings,
                        },
                    )
                    try:
                        cleaned_findings: list[dict] = []
                        for finding in single_request_quote_findings:
                            quote_text = str(finding.get("quotes", "")).strip()
                            quote_page = finding.get("page", None)
                            if not quote_text:
                                continue
                            cleaned_findings.append({"quotes": quote_text, "page": quote_page})

                        if cleaned_findings:
                            content = json.dumps(
                                {"quote_findings": cleaned_findings},
                                ensure_ascii=False,
                                indent=2,
                            )
                            await process.create_artifact(
                                mimetype="application/json",
                                description=(
                                    "Quote findings (single request over all pages) "
                                    f"[{len(cleaned_findings)} quotes]"
                                ),
                                content=(content + "\n").encode("utf-8"),
                                metadata={
                                    "quote_findings": cleaned_findings,
                                },
                            )
                    except Exception as art_exc:
                        await process.log(
                            "Warning: Failed to create aggregated single-request quote findings artifact: "
                            f"{art_exc}",
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

    async def _summarize_pdf_with_llm(
        self,
        process: IChatBioAgentProcess,
        pdf_url: str | None = None,
        text_source_path: str | None = None,
        text_content: str | None = None,
    ) -> Dict[str, str]:
        await process.log("Summarizing PDF content (title/abstract/methods/conclusion)...")

        max_chars = int(os.getenv("PDF_SUMMARY_MAX_CHARS", "20000"))
        snippet = ""
        if text_source_path and os.path.isfile(text_source_path):
            with open(text_source_path, "r", encoding="utf-8", errors="replace") as tf:
                snippet = tf.read(max_chars)
        elif text_content is not None:
            snippet = text_content[:max_chars]

        system_message = (
            "You are an assistant that reads scientific or technical PDF text and "
            "extracts four key sections: Title, Abstract, Methods, and Conclusion. "
            "Always answer in strict JSON with keys: "
            '\"title\", \"abstract\", \"methods\", \"conclusion\". '
            "If some information is not present, use an empty string for that field."
        )

        user_message = (
            "Read the following PDF content and provide a concise scientific summary "
            "with the following fields:\n"
            "- Title: the main title of the work\n"
            "- Abstract: a short overview of the problem, approach, and key results\n"
            "- Methods: the main methodology, models, or experimental setup\n"
            "- Conclusion: the main findings, implications, or takeaways\n\n"
            "Return ONLY a JSON object, no extra text, in this form:\n"
            "{\n"
            '  \"title\": \"...\",\n'
            '  \"abstract\": \"...\",\n'
            '  \"methods\": \"...\",\n'
            '  \"conclusion\": \"...\"\n'
            "}\n\n"
            "Here is the PDF content:\n\n"
            f"{snippet}"
        )

        client = OpenAI()
        model_name = os.getenv("OPENAI_PDF_SUMMARY_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content or ""

            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start != -1 and json_end != -1 and json_end >= json_start:
                content = content[json_start : json_end + 1]

            parsed = json.loads(content)

            summary: Dict[str, str] = {
                "title": str(parsed.get("title", "")).strip(),
                "abstract": str(parsed.get("abstract", "")).strip(),
                "methods": str(parsed.get("methods", "")).strip(),
                "conclusion": str(parsed.get("conclusion", "")).strip(),
            }

            await process.log(
                "PDF content summarization complete.",
                data={"model": model_name},
            )
            return summary

        except Exception as e:
            await process.log(
                f"Warning: LLM summarization failed, falling back to empty summary: {str(e)}",
                data={"pdf_url": pdf_url, "model": model_name},
            )
            return {
                "title": "",
                "abstract": "",
                "methods": "",
                "conclusion": "",
            }


def create_app() -> Starlette:
    agent = PDFReaderAgent()
    app = build_agent_app(agent)
    return app
