"""
PDF reading utilities using pypdf library
Extracts PDF URLs from text, downloads PDFs, and extracts text content
"""

import os
import re
import csv
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("Warning: pypdf library not installed. Install with: pip install pypdf")

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF library not installed. Install with: pip install pymupdf")


def extract_pdf_urls_from_text(text: str) -> List[str]:

    pdf_url_patterns = [
        r'https?://[^\s<>"{}|\\^`\[\]]+\.pdf(?:\?[^\s<>"{}|\\^`\[\]]*)?',  # URLs ending in .pdf
        r'https?://[^\s<>"{}|\\^`\[\]]+/pdf/[^\s<>"{}|\\^`\[\]]+',  # URLs with /pdf/ in path
        r'https?://arxiv\.org/pdf/[^\s<>"{}|\\^`\[\]]+',  # ArXiv PDF URLs
        r'https?://[^\s<>"{}|\\^`\[\]]+\.pdf\b',  # URLs ending with .pdf (word boundary)
    ]
    
    found_urls = set()
    
    for pattern in pdf_url_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the URL (remove trailing punctuation that might not be part of URL)
            url = match.rstrip('.,;:!?)')
            # Verify it's actually a PDF URL
            if '.pdf' in url.lower() or '/pdf/' in url.lower():
                found_urls.add(url)
    
    # Also check for URLs in markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+\.pdf[^)]*)\)'
    markdown_matches = re.findall(markdown_link_pattern, text, re.IGNORECASE)
    for text_part, url in markdown_matches:
        if '.pdf' in url.lower() or '/pdf/' in url.lower():
            found_urls.add(url.rstrip('.,;:!?)'))
    
    return list(found_urls)


def download_pdf(url: str, output_path: str) -> str:
    if os.path.exists(output_path):
        print(f"PDF already exists: {output_path}")
        return output_path
    
    print(f"Downloading PDF from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"PDF downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        raise


def get_pdf_num_pages(pdf_path: str) -> int:
    if not PYPDF_AVAILABLE:
        raise ImportError(
            "pypdf library is not installed. "
            "Please install it with: pip install pypdf"
        )
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    return len(reader.pages)


def resolve_page_span(
    num_pages: int,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> Tuple[int, int]:
    if num_pages < 1:
        raise ValueError("PDF has no pages")
    start = max(1, int(start_page))
    if start > num_pages:
        raise ValueError(f"start_page {start} exceeds PDF page count {num_pages}")
    if end_page is not None:
        end = min(int(end_page), num_pages)
    else:
        end = num_pages
    if end < start:
        raise ValueError(f"end_page {end} is before start_page {start}")
    if max_pages is not None:
        cap = int(max_pages)
        if cap < 1:
            raise ValueError("max_pages must be at least 1 when set")
        end = min(end, start + cap - 1)
    return start, end


def _safe_name(value: str, fallback: str = "pdf") -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    return name[:120] if name else fallback

def extract_images_with_pymupdf(
    pdf_path: str,
    output_dir: str,
    source_name: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    if not PYMUPDF_AVAILABLE:
        raise ImportError(
            "PyMuPDF library is not installed. "
            "Please install it with: pip install pymupdf"
        )
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    safe_source = _safe_name(source_name, "pdf")
    images_dir = Path(output_dir) / safe_source / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_files: List[str] = []
    page_counts: Dict[int, int] = {}
    total_images = 0

    doc = fitz.open(pdf_path)
    try:
        first, last = resolve_page_span(doc.page_count, start_page, end_page, max_pages)
        for page_num in range(first, last + 1):
            page = doc[page_num - 1]
            images = page.get_images(full=True)
            if not images:
                continue
            page_counts[page_num] = len(images)
            for img_idx, img in enumerate(images, start=1):
                xref = img[0]
                img_data = doc.extract_image(xref)
                if not img_data:
                    continue
                ext = (img_data.get("ext") or "png").lower()
                image_name = f"page_{page_num:04d}_img_{img_idx:03d}.{ext}"
                image_path = images_dir / image_name
                image_path.write_bytes(img_data["image"])
                image_files.append(str(image_path))
                total_images += 1
    finally:
        doc.close()

    return {
        "image_count": total_images,
        "image_files": image_files,
        "images_by_page": page_counts,
        "output_dir": str(images_dir),
    }


def embedded_image_dimensions(path: str) -> Optional[Tuple[int, int]]:
    """
    Return (width, height) in pixels for a saved image file, or None if unknown.
    PyMuPDF's page.get_images order does not match "Figure 1" vs header icons;
    callers use dimensions to rank likely figures (large rasters) over tiny logos.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if w > 0 and h > 0:
                return (int(w), int(h))
    except Exception:
        pass
    if PYMUPDF_AVAILABLE:
        try:
            pix = fitz.Pixmap(path)
            try:
                if pix.width > 0 and pix.height > 0:
                    return (int(pix.width), int(pix.height))
            finally:
                del pix
        except Exception:
            pass
    return None


def rank_embedded_image_paths_for_figure_artifacts(
    paths: List[str],
    *,
    min_area_px: int = 15_000,
    min_short_side_px: int = 64,
) -> List[str]:
    """
    Reorder embedded PDF image paths so large content images rank ahead of small
    raster icons (journal marks, Scopus badges, ORCID, etc.).

    PyMuPDF enumerates images in arbitrary order; filename order (img_001, img_002)
    is not "Figure 1" vs "Figure 2". We rank by decoded pixel area (descending)
    and drop images below size thresholds when possible.

    If every image fails the threshold (unusual PDFs), falls back to all paths
    sorted by area descending so we still prefer the largest assets.
    """
    if not paths:
        return []

    def passes(w: int, h: int) -> bool:
        if w <= 0 or h <= 0:
            return False
        area = w * h
        short_side = min(w, h)
        return area >= min_area_px and short_side >= min_short_side_px

    # (area, w, h, path) — one dimension read per path
    entries: List[Tuple[int, int, int, str]] = []
    for p in paths:
        dim = embedded_image_dimensions(p)
        if dim:
            w, h = dim
            entries.append((w * h, w, h, p))
        else:
            entries.append((0, 0, 0, p))

    filtered = [
        (a, p) for a, w, h, p in entries if w > 0 and h > 0 and passes(w, h)
    ]
    if not filtered:
        # Nothing met the bar (e.g. all thumbnails); use largest-by-area fallback.
        filtered = [(a, p) for a, w, h, p in entries]

    filtered.sort(key=lambda t: t[0], reverse=True)
    # Stable unique paths preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for _, p in filtered:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Caption / label cues in running text (not exhaustive; covers common publishing styles).
_TABLE_CAPTION_LINE = re.compile(
    r"(?im)(?:^|[\n\r])\s*(?:"
    r"Supplementary\s+(?:Table|Tab\.?)\s*|"
    r"Extended\s+Data\s+Table\s*|"
    r"(?:Table|TAB\.?|Tab\.)\s*(?!of\s+contents\b)"
    r")"
    r"(?:[S]?\d+[A-Za-z]?|[IVXLC]+)?\s*[.:)\]]?",
)

_FIGURE_CAPTION_LINE = re.compile(
    r"(?im)(?:^|[\n\r])\s*(?:"
    r"Supplementary\s+(?:Figure|Fig\.?)\s*|"
    r"(?:Figure|FIG\.?|Fig\.?|Plate|Scheme|Chart|Diagram|Graph|Map|Illustration|Image|Photo)\s*"
    r")"
    r"(?:[S]?\d+[A-Za-z]?|[IVXLC]+)?\s*[.:)\]]?",
)


def find_table_figure_cue_pages(page_texts: Dict[int, str]) -> Tuple[Set[int], Set[int]]:
    table_pages: Set[int] = set()
    figure_pages: Set[int] = set()
    for page, raw in (page_texts or {}).items():
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw
        if _TABLE_CAPTION_LINE.search(text):
            table_pages.add(int(page))
        if _FIGURE_CAPTION_LINE.search(text):
            figure_pages.add(int(page))
    return table_pages, figure_pages


def find_pages_with_table_word(page_texts: Dict[int, str]) -> Set[int]:
    out: Set[int] = set()
    for page, raw in (page_texts or {}).items():
        if not isinstance(raw, str) or not raw.strip():
            continue
        for m in re.finditer(r"(?i)\btable\b", raw):
            lo, hi = m.span()
            snippet = raw[max(0, lo - 28) : min(len(raw), hi + 36)]
            if re.search(r"(?i)\btable\s+of\s+contents\b", snippet):
                continue
            out.add(int(page))
            break
    return out


def read_csv_bundle_for_page(table_files: List[str], page: int) -> str:
    needle = f"page_{page:04d}_table_"
    paths = sorted(p for p in (table_files or []) if needle in Path(p).name)
    parts: List[str] = []
    for p in paths:
        try:
            body = Path(p).read_text(encoding="utf-8", errors="replace")
            parts.append(f"--- {Path(p).name} ---\n{body}")
        except OSError:
            continue
    return "\n\n".join(parts)


def list_image_paths_for_page(image_files: List[str], page: int) -> List[str]:
    needle = f"page_{page:04d}_img_"
    return sorted(p for p in (image_files or []) if needle in Path(p).name)


def render_pdf_page_to_png_bytes(
    pdf_path: str,
    page_1based: int,
    max_side_px: int = 1200,
) -> bytes:
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is required for page rendering.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        if page_1based < 1 or page_1based > doc.page_count:
            raise ValueError(f"page_1based {page_1based} out of range 1..{doc.page_count}")
        page = doc[page_1based - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        w, h = pix.width, pix.height
        if max_side_px > 0 and max(w, h) > max_side_px:
            scale = max_side_px / float(max(w, h))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def read_pdf_with_pypdf(
    pdf_path: str,
    strategy: str = "auto",
    include_page_breaks: bool = False,
    infer_table_structure: bool = True,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
    text_output_path: Optional[str] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], int]:

    if not PYPDF_AVAILABLE:
        raise ImportError(
            "pypdf library is not installed. "
            "Please install it with: pip install pypdf"
        )

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Parsing PDF: {pdf_path}")

    text_file = None
    if text_output_path:
        Path(text_output_path).parent.mkdir(parents=True, exist_ok=True)
        text_file = open(text_output_path, "w", encoding="utf-8")

    char_count = 0
    written_any = False

    def _write_fragment(s: str) -> None:
        nonlocal char_count, written_any
        if not text_file:
            return
        if written_any:
            text_file.write("\n\n")
            char_count += 2
        text_file.write(s)
        char_count += len(s)
        written_any = True

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF has {num_pages} pages")
        first, last = resolve_page_span(num_pages, start_page, end_page, max_pages)
        if (start_page, end_page, max_pages) != (1, None, None):
            print(f"Processing pages {first}-{last} (inclusive) of {num_pages}")

        elements: List[Dict[str, Any]] = []
        for page_num in range(first, last + 1):
            page = reader.pages[page_num - 1]
            try:
                text = page.extract_text()
                if text and text.strip():
                    stripped = text.strip()
                    element = {
                        "type": "Text",
                        "text": stripped,
                        "page_number": page_num,
                        "metadata": {
                            "page_number": page_num,
                            "total_pages": num_pages,
                            "extract_first_page": first,
                            "extract_last_page": last,
                        },
                    }
                    elements.append(element)
                    _write_fragment(stripped)

                    if include_page_breaks and page_num < last:
                        br = f"\n--- Page {page_num + 1} ---\n"
                        elements.append(
                            {
                                "type": "PageBreak",
                                "text": br,
                                "page_number": page_num + 1,
                                "metadata": {"page_number": page_num + 1},
                            }
                        )
                        _write_fragment(br.strip())
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page_num}: {e}")
                continue

        print(f"Successfully extracted {len(elements)} elements from PDF (page span {first}-{last})")
        if not text_file:
            char_count = len(extract_text_from_elements(elements))

        return elements, char_count

    except ValueError:
        raise
    except Exception as e:
        print(f"Error parsing PDF with pypdf: {e}")
        return None, 0
    finally:
        if text_file:
            text_file.close()

def read_pdf_with_unstructured(
    pdf_path: str,
    strategy: str = "auto",
    include_page_breaks: bool = False,
    infer_table_structure: bool = True,
) -> Optional[List]:
    elements, _ = read_pdf_with_pypdf(pdf_path, strategy, include_page_breaks, infer_table_structure)
    return elements


def extract_text_from_elements(elements: List) -> str:
    text_parts = []
    for element in elements:
        if isinstance(element, dict):
            text = element.get('text', '')
        elif hasattr(element, 'text'):
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
        if isinstance(element, dict):
            element_type = element.get('type', 'Unknown')
        else:
            element_type = type(element).__name__
        
        element_types.append(element_type)
        element_stats[element_type] = element_stats.get(element_type, 0) + 1
    
    return {
        'total_elements': len(elements),
        'element_types': element_stats,
        'all_types': element_types
    }


def save_content(content: str, output_dir: str, filename: str = "extracted_content.txt"):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Content saved to {output_path}")
    return output_path


def save_elements_json(elements: List, output_dir: str, filename: str = "elements.json"):

    try:
        import json
        
        # Convert elements to dictionary format
        elements_data = []
        for element in elements:
            element_dict = {
                'type': type(element).__name__,
                'text': getattr(element, 'text', str(element)),
            }
            
            # Add metadata if available
            if hasattr(element, 'metadata'):
                element_dict['metadata'] = element.metadata.__dict__ if hasattr(element.metadata, '__dict__') else str(element.metadata)
            
            elements_data.append(element_dict)
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(elements_data, f, indent=2, ensure_ascii=False)
        
        print(f"Elements JSON saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Warning: Could not save JSON: {e}")
        return None