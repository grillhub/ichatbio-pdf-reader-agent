"""Pydantic models, constants, and small helpers for the PDF reader agent."""

from typing import Dict, List, Literal, Optional

from ichatbio.types import Artifact
from pydantic import BaseModel, ConfigDict, Field

SECTION_KEYS = ("title", "abstract", "methods", "conclusion")

DESCRIPTION = """\
This agent can read and extract information from PDF documents. It:
- Extracts PDF URLs from user messages
- Downloads PDF files from URLs
- Extracts text content and structure from PDFs using advanced parsing
- Returns extracted information so iChatBio can answer questions about the PDF content

To use this agent, simply mention a PDF URL in your message. The agent will automatically detect it, download the PDF, and extract all text content for analysis.
"""


class PDFReaderParams(BaseModel):
    pdf_url: Optional[str] = Field(
        default=None,
        description="Direct URL to a PDF file. If not provided, URLs will be extracted from the request message.",
    )
    library: str = Field(
        default="pypdf",
        description="PDF reading library: 'pypdf' (default) or 'unstructured'.",
    )
    strategy: str = Field(
        default="fast",
        description="Parsing strategy. For unstructured: 'auto', 'hi_res', 'ocr_only', 'fast'. For pypdf: unused.",
    )
    include_page_breaks: bool = Field(default=False)
    infer_table_structure: bool = Field(default=True)
    pdf_artifact: Optional[Artifact] = Field(
        default=None,
        description="Artifact containing a PDF file to read instead of a URL.",
    )
    extract_title: bool = Field(default=True)
    extract_abstract: bool = Field(default=True)
    extract_methods: bool = Field(default=True)
    extract_conclusion: bool = Field(default=True)


class LangChainPageBlock(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(default="", description="Verbatim text for this block")
    block_type: Literal["heading", "table", "image", "text"] = Field(
        alias="type",
        description="Semantic type of this block",
    )
    keyword: List[str] = Field(default_factory=list)


class LangChainPageBlocks(BaseModel):
    blocks: List[LangChainPageBlock] = Field(default_factory=list)


def wanted_section_keys(params: Optional[PDFReaderParams]) -> list[str]:
    if params is None:
        return list(SECTION_KEYS)
    out: list[str] = []
    if params.extract_title:
        out.append("title")
    if params.extract_abstract:
        out.append("abstract")
    if params.extract_methods:
        out.append("methods")
    if params.extract_conclusion:
        out.append("conclusion")
    return out


def empty_sections_dict() -> Dict[str, str]:
    return {k: "" for k in SECTION_KEYS}
