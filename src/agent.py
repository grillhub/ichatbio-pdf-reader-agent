"""
PDF Reader iChatBio agent.

Flow: ``run`` → entrypoint LangChain graph → ``run_pdf_read_workflow`` → ``_handle_read_pdf``
→ pipeline graph (read / structure / summarize tools) → inner summary graph. See ``pdf_langchain_tools``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, override

import httpx
from pydantic import BaseModel
from starlette.applications import Starlette

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact

from .context import current_context
from .pdf_langchain_tools import (
    _get_pdf_entrypoint_langchain_graph,
    _get_pdf_pipeline_langchain_graph,
)
from .pdf_models import DESCRIPTION, PDFReaderParams, empty_sections_dict
from .pdf_reader import download_pdf, extract_pdf_urls_from_text
from .pdf_sessions import (
    PDFPipelineSession,
    _pdf_pipeline_session,
    _pdf_reader_agent,
    _pdf_run_params,
    _pdf_run_request,
)


class PDFReaderAgent(IChatBioAgent):
    def __init__(self) -> None:
        super().__init__()
        self.langchain_agent = _get_pdf_entrypoint_langchain_graph()

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="PDF Reader Agent",
            description=(
                "Reads and extracts information from PDF documents. "
                "Detects PDF URLs in messages, downloads them, and extracts text for analysis."
            ),
            icon=(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/"
                "PDF_file_icon.svg/1200px-PDF_file_icon.svg.png"
            ),
            entrypoints=[
                AgentEntrypoint(
                    id="read_pdf",
                    description=DESCRIPTION,
                    parameters=PDFReaderParams,
                )
            ],
        )

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: BaseModel,
    ) -> None:
        current_context.set(context)
        _pdf_reader_agent.set(self)
        _pdf_run_request.set(request)
        _pdf_run_params.set(params)
        await self.langchain_agent.ainvoke(
            {"messages": [{"role": "user", "content": request}]}
        )

    async def _handle_read_pdf(
        self,
        context: ResponseContext,
        request: str,
        params: Optional[PDFReaderParams],
    ) -> None:
        async with context.begin_process(
            summary="Reading and extracting information from PDF"
        ) as process:
            pdf_sources = await self._resolve_pdf_sources(params, request, process)
            if not pdf_sources:
                await context.reply(
                    "Error: No PDF source found. Please provide a PDF artifact or a PDF URL "
                    "in your message or as a parameter."
                )
                return

            all_results: List[dict] = []
            temp_dir: Optional[str] = None
            try:
                temp_dir = tempfile.mkdtemp(prefix="pdf_reader_")
                for idx, source in enumerate(pdf_sources):
                    try:
                        result = await self._process_one_pdf(
                            source=source,
                            idx=idx,
                            temp_dir=temp_dir,
                            pdf_sources=pdf_sources,
                            params=params,
                            process=process,
                        )
                        all_results.append(result)
                    except Exception as e:
                        err = str(e)
                        await process.log(f"Error processing PDF: {err}")
                        if source["kind"] == "url":
                            failed_url = source["url"]
                        else:
                            failed_url = f"artifact:{source['artifact'].local_id}"
                        all_results.append(
                            {"url": failed_url, "success": False, "error": err}
                        )

                await self._reply_processing_summary(context, all_results)
            except Exception as e:
                await process.log(f"Unexpected error: {e}")
                await context.reply(f"An error occurred while processing PDFs: {e}")
            finally:
                if temp_dir and os.path.isdir(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except OSError as e:
                        await process.log(
                            f"Warning: Failed to clean up temporary directory: {e}"
                        )

    async def _resolve_pdf_sources(
        self,
        params: Optional[PDFReaderParams],
        request: str,
        process: IChatBioAgentProcess,
    ) -> List[Dict]:
        sources: List[Dict] = []
        if params and params.pdf_artifact is not None:
            sources.append({"kind": "artifact", "artifact": params.pdf_artifact})
            await process.log(
                f"Using PDF artifact from parameters: local_id={params.pdf_artifact.local_id}"
            )
            return sources

        urls: List[str] = []
        if params and params.pdf_url:
            urls.append(params.pdf_url)
            await process.log(f"Downloading PDF from {params.pdf_url}")
        else:
            found = extract_pdf_urls_from_text(request)
            urls.extend(found)
            if found:
                await process.log(
                    f"Extracted {len(found)} PDF URL(s) from message: {', '.join(found)}"
                )

        for url in urls:
            sources.append({"kind": "url", "url": url})
        return sources

    async def _process_one_pdf(
        self,
        source: Dict,
        idx: int,
        temp_dir: str,
        pdf_sources: List[Dict],
        params: Optional[PDFReaderParams],
        process: IChatBioAgentProcess,
    ) -> dict:
        if source["kind"] == "artifact":
            artifact: Artifact = source["artifact"]
            pdf_path = os.path.join(
                temp_dir, f"artifact_{artifact.local_id or idx + 1}.pdf"
            )
            downloaded_path, effective_url = await self._download_pdf_from_artifact(
                artifact=artifact, output_path=pdf_path, process=process
            )
            pdf_url = effective_url or f"artifact:{artifact.local_id}"
            await process.log("PDF downloaded successfully from artifact.")
        else:
            pdf_url = source["url"]
            pdf_path = os.path.join(temp_dir, f"pdf_{idx + 1}.pdf")
            downloaded_path = download_pdf(pdf_url, pdf_path)
            await process.log("PDF downloaded successfully!")

        pref_library = (params.library if params else "pypdf") or "pypdf"
        pref_library = pref_library.lower().strip()
        if pref_library not in ("pypdf", "unstructured"):
            await process.log(
                f"Warning: Unknown library '{pref_library}' in params; orchestrator will choose."
            )
            pref_library = "pypdf"

        strategy = params.strategy if params else "fast"
        include_page_breaks = params.include_page_breaks if params else False
        infer_table_structure = params.infer_table_structure if params else True

        pipeline_session = PDFPipelineSession(
            process=process,
            params=params,
            downloaded_path=downloaded_path,
            pdf_url=pdf_url,
            strategy=strategy,
            include_page_breaks=include_page_breaks,
            infer_table_structure=infer_table_structure,
        )
        pl_tok = _pdf_pipeline_session.set(pipeline_session)
        try:
            pipe = _get_pdf_pipeline_langchain_graph()
            msg = (
                f"Process this PDF.\nSource: {pdf_url}\n"
                f"Preferred parsing library: {pref_library}\nStrategy: {strategy}\n"
                "Call read_pdf (use the preferred library unless you must switch), then "
                "structure_pdf_content, then summarize_pdf, then finish."
            )
            await pipe.ainvoke({"messages": [{"role": "user", "content": msg}]})
        finally:
            _pdf_pipeline_session.reset(pl_tok)

        if pipeline_session.pipeline_aborted:
            return {
                "url": pdf_url,
                "success": False,
                "error": pipeline_session.abort_reason or "PDF pipeline aborted",
            }

        if not pipeline_session.elements or not (pipeline_session.text_content or "").strip():
            await process.log(f"Warning: Pipeline did not produce text for PDF {pdf_url}")
            return {
                "url": pdf_url,
                "success": False,
                "error": "Pipeline incomplete: read_pdf did not succeed",
            }

        stats = pipeline_session.stats or {}
        text_content = pipeline_session.text_content or ""
        library = pipeline_session.library or pref_library
        structured_blocks = pipeline_session.structured_blocks or []
        structured_document = pipeline_session.structured_document or {
            "fileName": os.path.basename(downloaded_path),
            "pages": [],
        }
        sections_summary = pipeline_session.sections_summary or empty_sections_dict()

        await process.log(
            f"PDF pipeline complete: {stats.get('total_elements', 0)} elements, "
            f"{len(text_content)} characters, library={library}",
            data={
                "total_elements": stats.get("total_elements"),
                "element_types": stats.get("element_types"),
                "library": library,
                "pdf_url": pdf_url,
            },
        )

        result = {
            "url": pdf_url,
            "success": True,
            "library": library,
            "total_elements": stats.get("total_elements", 0),
            "element_types": stats.get("element_types", {}),
            "text_length": len(text_content),
            "text_content": text_content,
            "strategy": strategy,
            "structured_blocks": structured_blocks,
            "structured_document": structured_document,
            "sections_summary": sections_summary,
        }

        await self._emit_pdf_artifacts(
            process=process,
            pdf_url=pdf_url,
            idx=idx,
            pdf_sources=pdf_sources,
            text_content=text_content,
            stats=stats,
            strategy=strategy,
            library=library,
            structured_blocks=structured_blocks,
            structured_document=structured_document,
        )
        return result

    async def _emit_pdf_artifacts(
        self,
        process: IChatBioAgentProcess,
        pdf_url: str,
        idx: int,
        pdf_sources: List[Dict],
        text_content: str,
        stats: dict,
        strategy: str,
        library: str,
        structured_blocks: list,
        structured_document: dict,
    ) -> None:
        n = len(pdf_sources)
        suffix = f" (PDF {idx + 1} of {n})" if n > 1 else ""

        await process.create_artifact(
            mimetype="text/plain",
            description=f"Extracted text content from PDF: {pdf_url}{suffix}",
            content=text_content.encode("utf-8"),
            metadata={
                "source_url": pdf_url,
                "total_elements": stats.get("total_elements"),
                "element_types": stats.get("element_types"),
                "text_length": len(text_content),
                "strategy": strategy,
                "library": library,
                "pdf_index": idx + 1,
                "total_pdfs": n,
            },
        )

        try:
            await process.create_artifact(
                mimetype="application/json",
                description=f"Structured content blocks from PDF: {pdf_url}{suffix}",
                content=json.dumps(
                    structured_blocks, ensure_ascii=False, indent=2
                ).encode("utf-8"),
                metadata={
                    "source_url": pdf_url,
                    "total_elements": stats.get("total_elements"),
                    "element_types": stats.get("element_types"),
                    "text_length": len(text_content),
                    "strategy": strategy,
                    "library": library,
                    "pdf_index": idx + 1,
                    "total_pdfs": n,
                    "schema": "structured_blocks_v1",
                },
            )
        except Exception as e:
            await process.log(f"Warning: structured blocks artifact failed: {e}")

        try:
            await process.create_artifact(
                mimetype="application/json",
                description=f"LangChain-structured pages from PDF: {pdf_url}{suffix}",
                content=json.dumps(
                    structured_document, ensure_ascii=False, indent=2
                ).encode("utf-8"),
                metadata={
                    "source_url": pdf_url,
                    "library": library,
                    "pdf_index": idx + 1,
                    "total_pdfs": n,
                    "schema": "structured_document_langchain_v1",
                },
            )
        except Exception as e:
            await process.log(f"Warning: structured_document artifact failed: {e}")

    async def _reply_processing_summary(
        self, context: ResponseContext, all_results: List[dict]
    ) -> None:
        ok = sum(1 for r in all_results if r.get("success"))
        fail = len(all_results) - ok
        lines = [
            "**PDF Reading Complete**\n",
            f"**Total PDFs Processed:** {len(all_results)}",
            f"**Successful:** {ok}",
        ]
        if fail:
            lines.append(f"**Failed:** {fail}")
        lines.append("")

        for i, result in enumerate(all_results):
            lines.append(f"**PDF {i + 1}:** {result['url']}")
            if result.get("success"):
                lines.append(f"  - Library used: {result.get('library', 'pypdf')}")
                lines.append(
                    f"  - Elements extracted: {result.get('total_elements', 0)}"
                )
                lines.append(
                    f"  - Text length: {result.get('text_length', 0):,} characters"
                )
                et = result.get("element_types") or {}
                if et:
                    ts = ", ".join(f"{k} ({v})" for k, v in et.items())
                    lines.append(f"  - Element types: {ts}")
                sec = result.get("sections_summary") or {}
                if sec.get("title"):
                    lines.append(f"  - Inferred title: **{sec['title']}**")
                if sec.get("abstract"):
                    lines.extend(["  - Abstract (heuristic):", f"    {sec['abstract']}"])
                if sec.get("methods"):
                    lines.extend(["  - Methods (heuristic):", f"    {sec['methods']}"])
                if sec.get("conclusion"):
                    lines.extend(
                        ["  - Inferred conclusion:", f"    {sec['conclusion']}"]
                    )
            else:
                lines.append(f"  - Error: {result.get('error', 'Unknown error')}")
            lines.append("")

        lines.extend(
            [
                "The extracted text content has been saved as an artifact. "
                "You can now ask more detailed questions about the PDF content.",
            ]
        )
        await context.reply("\n".join(lines))

    async def _download_pdf_from_artifact(
        self,
        artifact: Artifact,
        output_path: str,
        process: IChatBioAgentProcess,
    ) -> tuple[str, Optional[str]]:
        if os.path.exists(output_path):
            await process.log(f"PDF already exists at {output_path}, skipping download.")
            urls = list(artifact.get_urls())
            return output_path, (urls[0] if urls else None)

        urls = list(artifact.get_urls())
        if not urls:
            await process.log(f"Artifact {artifact.local_id} has no retrievable URLs.")
            raise ValueError("Artifact has no URLs to download from.")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            last_error: Optional[Exception] = None
            for url in urls:
                try:
                    await process.log(
                        f"Downloading PDF artifact {artifact.local_id} from {url}"
                    )
                    if "localhost" in url:
                        url = url.replace("localhost", "98.91.29.140")
                    resp = await client.get(url)
                    if resp.is_success:
                        with open(output_path, "wb") as f:
                            f.write(resp.content)
                        await process.log(
                            f"Downloaded artifact {artifact.local_id} to {output_path}"
                        )
                        return output_path, url
                    await process.log(
                        f"Failed to download from {url}: {resp.status_code} {resp.reason_phrase}"
                    )
                except Exception as e:
                    last_error = e
                    await process.log(
                        f"Error downloading artifact {artifact.local_id} from {url}: {e}"
                    )

        raise ValueError(f"Failed to download artifact {artifact.local_id}") from last_error


def create_app() -> Starlette:
    return build_agent_app(PDFReaderAgent())
