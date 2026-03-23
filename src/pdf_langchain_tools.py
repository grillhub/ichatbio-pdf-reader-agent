"""
LangChain @tool definitions and create_agent graphs for entrypoint, pipeline, and section summary.
"""

import json
import os
import time
from typing import Optional

import langchain.agents
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime

from ichatbio.agent_response import IChatBioAgentProcess

from .context import current_context
from .pdf_helpers import (
    build_structured_blocks,
    collect_page_texts,
    structure_pdf_document_llm,
)
from .pdf_models import (
    PDFReaderParams,
    SECTION_KEYS,
    empty_sections_dict,
    wanted_section_keys,
)
from .pdf_prompts import (
    PDF_ENTRYPOINT_SYSTEM_PROMPT,
    PDF_PIPELINE_SYSTEM_PROMPT,
    PDF_SUMMARY_SYSTEM_PROMPT,
)
from .pdf_reader import (
    analyze_elements,
    download_pdf,
    extract_text_from_elements,
    read_pdf_with_pypdf,
)
from .pdf_reader_unstructured import (
    analyze_elements as analyze_elements_unstructured,
    extract_text_from_elements as extract_text_from_elements_unstructured,
    read_pdf_with_unstructured as read_pdf_with_unstructured_lib,
)
from .pdf_sessions import (
    PDFExtractionSession,
    PDFPipelineSession,
    _pdf_extraction_session,
    _pdf_pipeline_session,
    _pdf_reader_agent,
    _pdf_run_params,
    _pdf_run_request,
)

# --- cached graphs ---------------------------------------------------------------------------

_PDF_LANGCHAIN_SUMMARY_AGENT = None
_PDF_PIPELINE_LANGCHAIN_AGENT = None
_PDF_ENTRYPOINT_LANGCHAIN_AGENT = None


def _tool_context_error() -> Optional[str]:
    if current_context.get() is None:
        return "Error: no active iChatBio response context (current_context)."
    return None


# --- inner summary tools ---------------------------------------------------------------------


@tool
async def get_pdf_text_slice(offset: int, max_chars: int, runtime: ToolRuntime) -> str:
    """Read a slice of extracted PDF plain text (offset = char index, max_chars capped)."""
    err = _tool_context_error()
    if err:
        return err
    session = _pdf_extraction_session.get()
    if session is None:
        return "Error: no active PDF extraction session."
    text = session.working_text
    off = max(0, int(offset))
    cap = min(max(200, int(max_chars)), 12_000)
    end = min(len(text), off + cap)
    chunk = text[off:end]
    header = (
        f"[offset={off}, returned_chars={len(chunk)}, "
        f"working_text_total={len(text)}, full_pdf_chars={session.full_text_length}]\n"
    )
    return header + chunk


@tool
async def record_extracted_section(field: str, content: str, runtime: ToolRuntime) -> str:
    """Store one section: title, abstract, methods, or conclusion."""
    err = _tool_context_error()
    if err:
        return err
    session = _pdf_extraction_session.get()
    if session is None:
        return "Error: no active PDF extraction session."
    key = (field or "").strip().lower()
    allowed = set(SECTION_KEYS)
    if key not in allowed:
        return f"Invalid field {field!r}. Use one of: {sorted(allowed)}."
    if key not in session.fields_wanted:
        return f"Field {key!r} was not requested; skip it."
    session.sections[key] = (content or "").strip()
    await session.process.log(
        f"LangChain recorded section {key!r} ({len(session.sections[key])} chars)."
    )
    return f"Stored {key}."


@tool
async def list_pending_sections(runtime: ToolRuntime) -> str:
    """JSON list of section keys still required but empty."""
    err = _tool_context_error()
    if err:
        return json.dumps({"error": err})
    session = _pdf_extraction_session.get()
    if session is None:
        return "[]"
    pending = [
        f
        for f in session.fields_wanted
        if not (session.sections.get(f) or "").strip()
    ]
    return json.dumps(
        {"pending": pending, "requested": session.fields_wanted}, ensure_ascii=False
    )


@tool("finish", return_direct=True)
async def summary_loop_finish(message: str, runtime: ToolRuntime) -> str:
    """End the inner summarization loop after sections are filled."""
    session = _pdf_extraction_session.get()
    if session is None:
        return message
    session.finished = True
    await session.process.log(
        f"LangChain PDF summarization finished: {message}",
        data={"tool": "finish"},
    )
    return message


@tool("abort", return_direct=True)
async def summary_loop_abort(reason: str, runtime: ToolRuntime) -> str:
    """Abort the inner summarization loop."""
    session = _pdf_extraction_session.get()
    if session is None:
        return reason
    session.aborted = True
    session.abort_reason = reason
    await session.process.log(
        f"LangChain PDF summarization aborted: {reason}",
        data={"tool": "abort"},
    )
    return reason


def _get_pdf_summary_langchain_graph():
    global _PDF_LANGCHAIN_SUMMARY_AGENT
    if _PDF_LANGCHAIN_SUMMARY_AGENT is None:
        model_name = (
            os.getenv("OPENAI_PDF_SUMMARY_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        _PDF_LANGCHAIN_SUMMARY_AGENT = langchain.agents.create_agent(
            model=ChatOpenAI(
                model=model_name,
                temperature=0.2,
                tool_choice="required",
            ),
            tools=[
                get_pdf_text_slice,
                record_extracted_section,
                list_pending_sections,
                summary_loop_finish,
                summary_loop_abort,
            ],
            system_prompt=PDF_SUMMARY_SYSTEM_PROMPT,
        )
    return _PDF_LANGCHAIN_SUMMARY_AGENT


async def run_section_summary_loop(
    text_content: str,
    process: IChatBioAgentProcess,
    params: Optional[PDFReaderParams],
    pdf_url: str | None,
) -> dict[str, str]:
    """Run the inner LangChain tool loop for title / abstract / methods / conclusion."""
    wanted = wanted_section_keys(params)
    base = empty_sections_dict()
    if not wanted:
        await process.log("Skipping LangChain summarization (no sections requested).")
        return base

    await process.log(
        "Summarizing PDF with LangChain tool loop "
        f"({', '.join(wanted)})...",
        data={"pdf_url": pdf_url},
    )

    max_chars = int(os.getenv("PDF_SUMMARY_MAX_CHARS", "20000"))
    working = text_content[:max_chars]
    sections = {k: "" for k in wanted}
    session = PDFExtractionSession(
        process=process,
        working_text=working,
        full_text_length=len(text_content),
        fields_wanted=wanted,
        sections=sections,
    )
    token = _pdf_extraction_session.set(session)
    model_name = (
        os.getenv("OPENAI_PDF_SUMMARY_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )
    fields_str = ", ".join(wanted)
    user_message = (
        f"PDF source: {pdf_url or 'unknown'}\n"
        f"Extract these sections: {fields_str}.\n"
        f"Working copy length (visible to get_pdf_text_slice): {len(working)} characters; "
        f"full extracted text length: {len(text_content)} characters.\n"
        "Read with get_pdf_text_slice as needed, record each section, then call finish."
    )
    try:
        graph = _get_pdf_summary_langchain_graph()
        await graph.ainvoke({"messages": [{"role": "user", "content": user_message}]})
        if not session.finished and not session.aborted:
            await process.log(
                "Warning: LangChain agent ended without finish/abort; using partial sections.",
                data={"model": model_name},
            )
        elif session.finished:
            await process.log(
                "LangChain PDF summarization complete.",
                data={"model": model_name},
            )
    except Exception as e:
        await process.log(
            f"Warning: LangChain summarization failed: {e}",
            data={"pdf_url": pdf_url, "model": model_name},
        )
    finally:
        _pdf_extraction_session.reset(token)

    return {k: (session.sections.get(k) or "").strip() for k in SECTION_KEYS}


# --- pipeline tools --------------------------------------------------------------------------


@tool
async def read_pdf(library: str, runtime: ToolRuntime, strategy: str = "fast") -> str:
    """Parse the PDF on disk (pypdf or unstructured)."""
    err = _tool_context_error()
    if err:
        return err
    ps = _pdf_pipeline_session.get()
    if ps is None:
        return "Error: no active PDF pipeline session."
    lib = (library or "").lower().strip()
    if lib not in ("pypdf", "unstructured"):
        return 'Error: library must be "pypdf" or "unstructured".'
    path = ps.downloaded_path
    if not path or not os.path.isfile(path):
        return "Error: PDF file is missing on disk."
    strat = (strategy or ps.strategy or "fast").strip() or "fast"
    start = time.perf_counter()
    try:
        if lib == "unstructured":
            elements = read_pdf_with_unstructured_lib(
                pdf_path=path,
                strategy=strat,
                include_page_breaks=ps.include_page_breaks,
                infer_table_structure=ps.infer_table_structure,
            )
        else:
            elements = read_pdf_with_pypdf(
                pdf_path=path,
                strategy=strat,
                include_page_breaks=ps.include_page_breaks,
                infer_table_structure=ps.infer_table_structure,
            )
    except Exception as e:
        return f"Error parsing PDF: {e}"
    elapsed = time.perf_counter() - start
    if not elements:
        return "Error: no elements extracted from PDF."
    if lib == "unstructured":
        text_content = extract_text_from_elements_unstructured(elements)
        stats = analyze_elements_unstructured(elements)
    else:
        text_content = extract_text_from_elements(elements)
        stats = analyze_elements(elements)
    ps.library = lib
    ps.elements = elements
    ps.text_content = text_content
    ps.stats = stats
    await ps.process.log(
        f"read_pdf: library={lib}, elements={stats.get('total_elements', 0)}, "
        f"chars={len(text_content)}, time={elapsed:.3f}s",
        data={"library": lib, "pdf_url": ps.pdf_url},
    )
    return (
        f"Parsed with {lib} in {elapsed:.3f}s: {stats.get('total_elements', 0)} elements, "
        f"{len(text_content)} characters. Next call structure_pdf_content."
    )


@tool
async def structure_pdf_content(runtime: ToolRuntime) -> str:
    """Heuristic blocks + LLM page structure. Requires read_pdf first."""
    err = _tool_context_error()
    if err:
        return err
    ps = _pdf_pipeline_session.get()
    if ps is None:
        return "Error: no active PDF pipeline session."
    if not ps.elements or not ps.library:
        return "Error: call read_pdf first."
    ps.structured_blocks = build_structured_blocks(ps.elements, ps.library)
    page_texts = collect_page_texts(ps.elements, ps.library)
    ps.structured_document = await structure_pdf_document_llm(
        file_name=os.path.basename(ps.downloaded_path),
        page_texts=page_texts,
        process=ps.process,
    )
    n_pages = len(page_texts)
    n_blocks = len((ps.structured_document or {}).get("pages") or [])
    return (
        f"Structured content: {len(ps.structured_blocks)} heuristic block(s), "
        f"{n_blocks} LLM segment(s) across {n_pages} page(s). Next call summarize_pdf."
    )


@tool
async def summarize_pdf(runtime: ToolRuntime) -> str:
    """Section extraction via inner LangChain loop. Requires read_pdf first."""
    err = _tool_context_error()
    if err:
        return err
    ps = _pdf_pipeline_session.get()
    if ps is None:
        return "Error: no active PDF pipeline session."
    if not (ps.text_content or "").strip():
        return "Error: call read_pdf first."
    ps.sections_summary = await run_section_summary_loop(
        ps.text_content, ps.process, ps.params, ps.pdf_url
    )
    wanted = wanted_section_keys(ps.params)
    filled = [k for k in wanted if (ps.sections_summary or {}).get(k, "").strip()]
    return (
        f"Section extraction done; filled: {filled or 'none'}. "
        f"If read_pdf and structure_pdf_content already succeeded, call finish."
    )


@tool("finish", return_direct=True)
async def finish(message: str, runtime: ToolRuntime) -> str:
    """End the pipeline successfully."""
    err = _tool_context_error()
    if err:
        return err
    ps = _pdf_pipeline_session.get()
    if ps is None:
        return message
    ps.pipeline_finished = True
    await ps.process.log(
        f"PDF pipeline finished: {message}",
        data={"tool": "pipeline_finish"},
    )
    return message


@tool("abort", return_direct=True)
async def abort(reason: str, runtime: ToolRuntime) -> str:
    """Abort the pipeline."""
    err = _tool_context_error()
    if err:
        return reason
    ps = _pdf_pipeline_session.get()
    if ps is None:
        return reason
    ps.pipeline_aborted = True
    ps.abort_reason = reason
    await ps.process.log(
        f"PDF pipeline aborted: {reason}",
        data={"tool": "pipeline_abort"},
    )
    return reason


def _get_pdf_pipeline_langchain_graph():
    global _PDF_PIPELINE_LANGCHAIN_AGENT
    if _PDF_PIPELINE_LANGCHAIN_AGENT is None:
        model_name = (
            os.getenv("OPENAI_PDF_PIPELINE_MODEL")
            or os.getenv("OPENAI_PDF_SUMMARY_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        _PDF_PIPELINE_LANGCHAIN_AGENT = langchain.agents.create_agent(
            model=ChatOpenAI(
                model=model_name,
                temperature=0.2,
                tool_choice="required",
            ),
            tools=[
                read_pdf,
                structure_pdf_content,
                summarize_pdf,
                finish,
                abort,
            ],
            system_prompt=PDF_PIPELINE_SYSTEM_PROMPT,
        )
    return _PDF_PIPELINE_LANGCHAIN_AGENT


# --- entrypoint tools ------------------------------------------------------------------------


@tool
async def run_pdf_read_workflow(runtime: ToolRuntime) -> str:
    """Full workflow: sources, download, pipeline, artifacts, summary reply."""
    ctx = current_context.get()
    if ctx is None:
        return "Error: current_context is not set."
    agent = _pdf_reader_agent.get()
    if agent is None:
        return "Error: PDF reader agent is not bound."
    request = _pdf_run_request.get() or ""
    raw = _pdf_run_params.get()
    prm = raw if isinstance(raw, PDFReaderParams) else None
    await agent._handle_read_pdf(ctx, request, prm)
    return "PDF workflow finished (summary and artifacts were sent on the iChatBio channel)."


@tool("finish", return_direct=True)
async def entrypoint_finish(message: str, runtime: ToolRuntime) -> str:
    """End entrypoint graph (no extra channel reply)."""
    return message


@tool("abort", return_direct=True)
async def entrypoint_abort(reason: str, runtime: ToolRuntime) -> str:
    """End entrypoint graph and notify user."""
    ctx = current_context.get()
    if ctx is not None:
        await ctx.reply(reason)
    return reason


def _get_pdf_entrypoint_langchain_graph():
    global _PDF_ENTRYPOINT_LANGCHAIN_AGENT
    if _PDF_ENTRYPOINT_LANGCHAIN_AGENT is None:
        model_name = (
            os.getenv("OPENAI_PDF_ENTRYPOINT_MODEL")
            or os.getenv("OPENAI_PDF_PIPELINE_MODEL")
            or os.getenv("OPENAI_PDF_SUMMARY_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        _PDF_ENTRYPOINT_LANGCHAIN_AGENT = langchain.agents.create_agent(
            model=ChatOpenAI(
                model=model_name,
                temperature=0.2,
                tool_choice="required",
            ),
            tools=[run_pdf_read_workflow, entrypoint_finish, entrypoint_abort],
            system_prompt=PDF_ENTRYPOINT_SYSTEM_PROMPT,
        )
    return _PDF_ENTRYPOINT_LANGCHAIN_AGENT
