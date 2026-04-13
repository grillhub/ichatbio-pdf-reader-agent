"""
PDF reading utilities using Unstructured library
Extracts PDF URLs from text, downloads PDFs, and extracts text content
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from .pdf_reader import get_pdf_num_pages, resolve_page_span

# Try to import Unstructured components
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from unstructured import partition_pdf, partition
        UNSTRUCTURED_AVAILABLE = True
    except ImportError:
        UNSTRUCTURED_AVAILABLE = False
        print("Warning: Unstructured library not installed. Install with: pip install 'unstructured[pdf]'")


def _element_page_number(element) -> int:
    meta = getattr(element, "metadata", None)
    if meta is None:
        return 1
    pn = getattr(meta, "page_number", None)
    if pn is None and isinstance(meta, dict):
        pn = meta.get("page_number")
    try:
        return max(int(pn), 1)
    except (TypeError, ValueError):
        return 1


def _stream_unstructured_text_to_file(elements: List, path: str) -> int:
    char_count = 0
    written = False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for el in elements:
            if hasattr(el, "text"):
                t = el.text
            else:
                t = str(el)
            if not t or not str(t).strip():
                continue
            s = str(t).strip()
            if written:
                f.write("\n\n")
                char_count += 2
            f.write(s)
            char_count += len(s)
            written = True
    return char_count


def read_pdf_with_unstructured(
    pdf_path: str,
    strategy: str = "auto",
    include_page_breaks: bool = False,
    infer_table_structure: bool = True,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
    text_output_path: Optional[str] = None,
) -> Tuple[Optional[List], int]:
    if not UNSTRUCTURED_AVAILABLE:
        raise ImportError(
            "Unstructured library is not installed. "
            "Please install it with: pip install 'unstructured[pdf]'"
        )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Parsing PDF with Unstructured: {pdf_path}")
    print(f"Using strategy: {strategy}")

    num_pages = get_pdf_num_pages(pdf_path)
    first, last = resolve_page_span(num_pages, start_page, end_page, max_pages)
    if (start_page, end_page, max_pages) != (1, None, None):
        print(f"Will keep elements on pages {first}-{last} (inclusive) of {num_pages}")

    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            include_page_breaks=include_page_breaks,
            infer_table_structure=infer_table_structure,
        )

        print(f"Successfully extracted {len(elements)} elements from PDF (before page filter)")
        filtered = [
            el for el in elements if first <= _element_page_number(el) <= last
        ]
        print(f"After page filter: {len(filtered)} elements")

        if text_output_path:
            char_count = _stream_unstructured_text_to_file(filtered, text_output_path)
        else:
            char_count = len(extract_text_from_elements(filtered))

        return filtered, char_count

    except Exception as e:
        print(f"Error parsing PDF with partition_pdf: {e}")
        print("Trying alternative method with partition.auto...")

        try:
            elements = partition(
                filename=pdf_path,
                strategy=strategy,
            )
            print(f"Successfully extracted {len(elements)} elements using partition.auto")
            filtered = [
                el for el in elements if first <= _element_page_number(el) <= last
            ]
            if text_output_path:
                char_count = _stream_unstructured_text_to_file(filtered, text_output_path)
            else:
                char_count = len(extract_text_from_elements(filtered))
            return filtered, char_count
        except Exception as e2:
            print(f"Error with partition.auto: {e2}")
            return None, 0


def extract_text_from_elements(elements: List) -> str:
    text_parts = []
    for element in elements:
        # Get text content from element
        if hasattr(element, 'text'):
            text = element.text
        elif hasattr(element, '__str__'):
            text = str(element)
        else:
            text = repr(element)
        
        if text and text.strip():
            text_parts.append(text.strip())
    
    return "\n\n".join(text_parts)


def analyze_elements(elements: List) -> dict:
    element_stats = {}
    element_types = []
    
    for element in elements:
        element_type = type(element).__name__
        element_types.append(element_type)
        element_stats[element_type] = element_stats.get(element_type, 0) + 1
    
    return {
        'total_elements': len(elements),
        'element_types': element_stats,
        'all_types': element_types
    }
