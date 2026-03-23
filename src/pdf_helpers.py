"""Pure PDF layout helpers and per-page LLM structuring (no LangChain agent graphs)."""

import asyncio
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ichatbio.agent_response import IChatBioAgentProcess

from .pdf_models import LangChainPageBlocks
from .pdf_prompts import PDF_STRUCTURE_SYSTEM_PROMPT

_STRUCTURE_BLOCK_TYPES = frozenset({"heading", "table", "image", "text"})


def normalize_structure_block_type(value: str) -> str:
    v = (value or "text").strip().lower()
    if v in _STRUCTURE_BLOCK_TYPES:
        return v
    return "text"


def _page_from_unstructured_meta(meta) -> int:
    if meta is None:
        return 1
    page = getattr(meta, "page_number", None)
    if page is None and isinstance(meta, dict):
        page = meta.get("page_number")
    if page is None:
        return 1
    try:
        return max(int(page), 1)
    except (TypeError, ValueError):
        return 1


def collect_page_texts(elements, library: str) -> List[Tuple[int, str]]:
    if not elements:
        return []
    lib = (library or "pypdf").lower().strip()

    if lib == "pypdf":
        pairs: List[Tuple[int, str]] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            if element.get("type") in ("PageBreak",):
                continue
            if element.get("type") != "Text":
                continue
            raw = (element.get("text") or "").strip()
            if not raw:
                continue
            meta = element.get("metadata") or {}
            try:
                pn = int(element.get("page_number") or meta.get("page_number") or 1)
            except (TypeError, ValueError):
                pn = 1
            pairs.append((max(pn, 1), raw))
        pairs.sort(key=lambda x: x[0])
        return pairs

    by_page: defaultdict[int, List[str]] = defaultdict(list)
    for element in elements or []:
        if isinstance(element, dict):
            continue
        meta = getattr(element, "metadata", None)
        pn = _page_from_unstructured_meta(meta)
        chunk = (getattr(element, "text", None) or "").strip()
        if chunk:
            by_page[pn].append(chunk)
    return [(p, "\n\n".join(parts)) for p, parts in sorted(by_page.items(), key=lambda x: x[0])]


def _page_from_block_metadata(meta) -> int:
    if meta is None:
        return 1
    page = getattr(meta, "page_number", None)
    if page is None and isinstance(meta, dict):
        page = meta.get("page_number")
    if page is None:
        return 1
    try:
        return max(int(page), 1)
    except (TypeError, ValueError):
        return 1


def build_structured_blocks(elements, library: str) -> list[dict]:
    structured: list[dict] = []
    for element in elements or []:
        if isinstance(element, dict):
            element_type = element.get("type", "Unknown")
            text = element.get("text", "") or ""
            page_meta = element.get("metadata", {})
            page_number = element.get("page_number") or page_meta.get("page_number") or 1
            page_idx = max(int(page_number), 1)
            if element_type == "Text" and text.strip():
                structured.append(
                    {"type": "text", "text": text.strip(), "page_number": page_idx}
                )
            continue

        element_type = type(element).__name__
        meta = getattr(element, "metadata", None)
        page_idx = _page_from_block_metadata(meta)

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

        if any(k in element_type.lower() for k in ("image", "figure", "picture", "photo")):
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
                {"type": "text", "text": text.strip(), "page_number": page_idx}
            )
    return structured


async def structure_pdf_document_llm(
    file_name: str,
    page_texts: List[Tuple[int, str]],
    process: IChatBioAgentProcess,
) -> Dict:
    pages_out: List[Dict] = []
    if not page_texts:
        return {"fileName": file_name, "pages": []}

    model_name = (
        os.getenv("OPENAI_PDF_STRUCTURE_MODEL")
        or os.getenv("OPENAI_PDF_SUMMARY_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )
    max_chars = int(os.getenv("PDF_STRUCTURE_MAX_PAGE_CHARS", "14000"))
    max_concurrent = max(1, int(os.getenv("PDF_STRUCTURE_MAX_CONCURRENT", "3")))

    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(LangChainPageBlocks)
    sem = asyncio.Semaphore(max_concurrent)

    async def one_page(page_num: int, page_plain: str) -> List[Dict]:
        if not (page_plain or "").strip():
            return []
        body = page_plain.strip()
        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars] + "\n\n[... truncated for structuring ...]"
        async with sem:
            try:
                parsed = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=PDF_STRUCTURE_SYSTEM_PROMPT),
                        HumanMessage(
                            content=(
                                f"Page number (1-based): {page_num}\n\nPage text:\n\n{body}"
                            )
                        ),
                    ]
                )
            except Exception as e:
                await process.log(
                    f"Warning: LangChain page structuring failed for page {page_num}: {e}",
                    data={"model": model_name, "page": page_num},
                )
                return [
                    {
                        "page": page_num,
                        "text": page_plain.strip()[:max_chars],
                        "type": "text",
                        "keyword": [],
                    }
                ]
        if isinstance(parsed, dict):
            parsed = LangChainPageBlocks.model_validate(parsed)
        rows: List[Dict] = []
        for b in parsed.blocks:
            t = (b.text or "").strip()
            if not t:
                continue
            rows.append(
                {
                    "page": page_num,
                    "text": t,
                    "type": normalize_structure_block_type(b.block_type or "text"),
                    "keyword": [
                        k.strip()
                        for k in (b.keyword or [])
                        if k is not None and str(k).strip()
                    ],
                }
            )
        if not rows and page_plain.strip():
            rows.append(
                {
                    "page": page_num,
                    "text": page_plain.strip()[:max_chars],
                    "type": "text",
                    "keyword": [],
                }
            )
        if truncated and rows:
            await process.log(
                f"Note: Page {page_num} was truncated to {max_chars} chars for structuring.",
                data={"page": page_num},
            )
        return rows

    for chunk in await asyncio.gather(*[one_page(pn, txt) for pn, txt in page_texts]):
        pages_out.extend(chunk)

    await process.log(
        f"LangChain structured document: {len(pages_out)} block(s) on {len(page_texts)} page(s).",
        data={"model": model_name, "file": file_name},
    )
    return {"fileName": file_name, "pages": pages_out}
