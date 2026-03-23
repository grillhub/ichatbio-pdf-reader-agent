from typing import override, Optional, List, Dict
import time
import json
import re
import tempfile
import os
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
    extract_text_from_elements,
    analyze_elements
)
from .pdf_reader_unstructured import (
    read_pdf_with_unstructured as read_pdf_with_unstructured_lib,
    extract_text_from_elements as extract_text_from_elements_unstructured,
    analyze_elements as analyze_elements_unstructured
)

DESCRIPTION = """\
This agent can read and extract information from PDF documents. It:
- Extracts PDF URLs from user messages
- Downloads PDF files from URLs
- Extracts text content and structure from PDFs using advanced parsing
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


class PDFReaderAgent(IChatBioAgent):
    """Agent that reads and extracts information from PDF documents"""

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

            # Determine PDF source(s) to process: either artifact, URL param, or URLs in request text
            pdf_sources: List[Dict] = []

            # Prefer explicit artifact when provided
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
                # Fallback to URL from parameters or message text
                pdf_urls: List[str] = []

                if params and isinstance(params, PDFReaderParams) and params.pdf_url:
                    pdf_urls.append(params.pdf_url)
                    await process.log(f"Downloading PDF from {params.pdf_url}")
                else:
                    # Extract URLs from request message
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

            # Process each PDF source
            all_results = []
            temp_dir = None

            try:
                # Create temporary directory for downloaded PDFs
                temp_dir = tempfile.mkdtemp(prefix="pdf_reader_")
                # await process.log(f"Created temporary directory: {temp_dir}")

                for idx, source in enumerate(pdf_sources):
                    # await process.log(f"Processing PDF {idx + 1}/{len(pdf_sources)}: {source}")

                    try:
                        # Download PDF from URL or artifact
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

                            # await process.log(f"Downloading PDF from {pdf_url}...")
                            downloaded_path = download_pdf(pdf_url, pdf_path)
                            await process.log("PDF downloaded successfully!")

                        # Get parsing parameters
                        library = params.library if params and isinstance(params, PDFReaderParams) else "pypdf"
                        strategy = params.strategy if params and isinstance(params, PDFReaderParams) else "fast"
                        include_page_breaks = params.include_page_breaks if params and isinstance(params, PDFReaderParams) else False
                        infer_table_structure = params.infer_table_structure if params and isinstance(params, PDFReaderParams) else True

                        # Normalize library name
                        library = library.lower().strip()
                        if library not in ["pypdf", "unstructured"]:
                            await process.log(f"Warning: Unknown library '{library}', defaulting to 'pypdf'")
                            library = "pypdf"

                        await process.log(f"Parsing PDF with {library}")

                        # Measure PDF extraction processing time
                        start_time = time.perf_counter()

                        # Read PDF with selected library
                        if library == "unstructured":
                            elements = read_pdf_with_unstructured_lib(
                                pdf_path=downloaded_path,
                                strategy="fast",
                                include_page_breaks=include_page_breaks,
                                infer_table_structure=infer_table_structure
                            )
                        else:
                            # Default to pypdf
                            elements = read_pdf_with_pypdf(
                                pdf_path=downloaded_path,
                                strategy=strategy,
                                include_page_breaks=include_page_breaks,
                                infer_table_structure=infer_table_structure
                            )

                        extraction_time = time.perf_counter() - start_time
                        await process.log(
                            f"PDF extraction processing time: {extraction_time:.3f} seconds",
                            data={
                                "processing_time_seconds": extraction_time,
                                "library": library,
                                "strategy": strategy,
                                "pdf_url": pdf_url,
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

                        # Extract text and analyze elements based on library
                        if library == "unstructured":
                            # Use unstructured-specific extractors for unstructured objects
                            text_content = extract_text_from_elements_unstructured(elements)
                            stats = analyze_elements_unstructured(elements)
                        else:
                            # Use pypdf extractors for dict-based elements
                            text_content = extract_text_from_elements(elements)
                            stats = analyze_elements(elements)

                        text_length = len(text_content)
                        await process.log(
                            f"Extracted {stats['total_elements']} elements and {text_length} characters of text from PDF",
                            data={
                                "total_elements": stats['total_elements'],
                                "element_types": stats['element_types'],
                                "library": library
                            }
                        )

                        # Build a structured representation of the PDF content
                        structured_blocks = self._build_structured_blocks(elements, library)

                        # Use LLM to derive high-level sections (title, abstract, methods, conclusion)
                        sections_summary = await self._summarize_pdf_with_llm(
                            text_content=text_content,
                            process=process,
                            pdf_url=pdf_url,
                        )

                        # Store result
                        result = {
                            "url": pdf_url,
                            "success": True,
                            "library": library,
                            "total_elements": stats['total_elements'],
                            "element_types": stats['element_types'],
                            "text_length": text_length,
                            "text_content": text_content,
                            "strategy": strategy,
                            "structured_blocks": structured_blocks,
                            "sections_summary": sections_summary,
                        }

                        all_results.append(result)

                        # Create artifact with extracted text
                        artifact_description = f"Extracted text content from PDF: {pdf_url}"
                        if len(pdf_sources) > 1:
                            artifact_description += f" (PDF {idx + 1} of {len(pdf_sources)})"

                        await process.create_artifact(
                            mimetype="text/plain",
                            description=artifact_description,
                            content=text_content.encode("utf-8"),
                            metadata={
                                "source_url": pdf_url,
                                "total_elements": stats['total_elements'],
                                "element_types": stats['element_types'],
                                "text_length": text_length,
                                "strategy": strategy,
                                "library": library,
                                "pdf_index": idx + 1,
                                "total_pdfs": len(pdf_sources)
                            }
                        )
                        # await process.log(f"Created artifact with extracted text content from PDF {idx + 1}")

                        # Create artifact with structured content blocks (text/image/table)
                        try:
                            structured_content_bytes = json.dumps(
                                structured_blocks,
                                ensure_ascii=False,
                                indent=2,
                            ).encode("utf-8")
                            structured_description = (
                                f"Structured content blocks (text/image/table) from PDF: {pdf_url}"
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
                            # await process.log(
                            #     f"Created artifact with structured content blocks from PDF {idx + 1}"
                            # )
                        except Exception as e:
                            await process.log(
                                f"Warning: Failed to create structured blocks artifact for PDF {idx + 1}: {str(e)}"
                            )

                    except Exception as e:
                        error_msg = str(e)
                        await process.log(f"Error processing PDF {pdf_url}: {error_msg}")
                        all_results.append({
                            "url": pdf_url,
                            "success": False,
                            "error": error_msg
                        })

                # Build summary response
                successful = sum(1 for r in all_results if r.get("success", False))
                failed = len(all_results) - successful

                summary = f"**PDF Reading Complete**\n\n"
                summary += f"**Total PDFs Processed:** {len(all_results)}\n"
                summary += f"**Successful:** {successful}\n"
                if failed > 0:
                    summary += f"**Failed:** {failed}\n"
                summary += "\n"

                # Add details for each PDF
                for idx, result in enumerate(all_results):
                    summary += f"**PDF {idx + 1}:** {result['url']}\n"
                    if result.get("success"):
                        summary += f"  - Library used: {result.get('library', 'pypdf')}\n"
                        summary += f"  - Elements extracted: {result.get('total_elements', 0)}\n"
                        summary += f"  - Text length: {result.get('text_length', 0):,} characters\n"
                        element_types = result.get('element_types', {})
                        if element_types:
                            types_str = ", ".join([f"{k} ({v})" for k, v in element_types.items()])
                            summary += f"  - Element types: {types_str}\n"
                        # Add high-level content summary (title, abstract, methods, conclusion)
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
        """
        Build a structured, human-readable representation of the PDF content.

        The structure is a list of blocks like:
        - {"type": "text", "text": "...", "page_number": 1}
        - {"type": "image", "img_path": "...", "image_caption": [...], "image_footnote": [], "page_number": 2}
        - {"type": "table", "table_body": "<table>...</table>", "table_caption": [...], "table_footnote": [], "page_number": 3}
        """
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
            # Keep 1-based page numbering
            return max(page_int, 1)

        for element in elements or []:
            # pypdf elements are simple dicts
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
                # For pypdf we currently do not distinguish images/tables, so we only emit text blocks.
                continue

            # Unstructured elements: infer type from class name
            element_type = type(element).__name__
            meta = getattr(element, "metadata", None)
            page_idx = _get_page_number_from_metadata(meta)

            # Table-like elements
            if "table" in element_type.lower():
                # Try to extract HTML representation if available, otherwise fall back to text
                table_html = None
                if meta is not None:
                    # Common unstructured metadata attributes
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
        """
        Download a PDF file from an artifact URL into a local path.

        Returns (downloaded_path, effective_url).
        """
        # If the file already exists, skip re-download
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
                    if "localhost" in url:
                        url = url.replace("localhost", "98.81.202.253")
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

    async def _summarize_pdf_with_llm(
        self,
        text_content: str,
        process: IChatBioAgentProcess,
        pdf_url: str | None = None,
    ) -> Dict[str, str]:
        """
        Use an LLM (via OpenAI-compatible API) to extract:
        - title
        - abstract
        - methods
        - conclusion
        from the PDF's extracted text content.
        """
        await process.log("Summarizing PDF content (title/abstract/methods/conclusion)...")

        # Truncate content to avoid exceeding model context limits
        max_chars = int(os.getenv("PDF_SUMMARY_MAX_CHARS", "20000"))
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

            # Ensure we only keep the JSON portion if the model adds any extra text accidentally
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start != -1 and json_end != -1 and json_end >= json_start:
                content = content[json_start : json_end + 1]

            parsed = json.loads(content)

            # Normalize and ensure all expected keys exist
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
            # Fallback: empty fields so the rest of the pipeline still works
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
