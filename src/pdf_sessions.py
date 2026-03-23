"""Dataclasses and ContextVars for PDF pipeline / summary / entrypoint state."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ichatbio.agent_response import IChatBioAgentProcess
from pydantic import BaseModel

from .pdf_models import PDFReaderParams


@dataclass
class PDFExtractionSession:
    """Inner LangChain loop: slice + record sections (title, abstract, …)."""

    process: IChatBioAgentProcess
    working_text: str
    full_text_length: int
    fields_wanted: list[str]
    sections: Dict[str, str] = field(default_factory=dict)
    finished: bool = False
    aborted: bool = False
    abort_reason: str = ""


@dataclass
class PDFPipelineSession:
    """Outer pipeline: read_pdf → structure → summarize tools."""

    process: IChatBioAgentProcess
    params: Optional[PDFReaderParams]
    downloaded_path: str
    pdf_url: str
    strategy: str
    include_page_breaks: bool
    infer_table_structure: bool
    library: Optional[str] = None
    elements: Optional[list] = None
    text_content: Optional[str] = None
    stats: Optional[Dict] = None
    structured_blocks: Optional[list] = None
    structured_document: Optional[Dict] = None
    sections_summary: Optional[Dict[str, str]] = None
    pipeline_finished: bool = False
    pipeline_aborted: bool = False
    abort_reason: str = ""


_pdf_extraction_session: ContextVar[Optional[PDFExtractionSession]] = ContextVar(
    "_pdf_extraction_session", default=None
)
_pdf_pipeline_session: ContextVar[Optional[PDFPipelineSession]] = ContextVar(
    "_pdf_pipeline_session", default=None
)
_pdf_reader_agent: ContextVar[Optional[Any]] = ContextVar("_pdf_reader_agent", default=None)
_pdf_run_request: ContextVar[str] = ContextVar("_pdf_run_request", default="")
_pdf_run_params: ContextVar[Optional[BaseModel]] = ContextVar("_pdf_run_params", default=None)
