"""
PDF reading utilities using Unstructured library
Extracts PDF URLs from text, downloads PDFs, and extracts text content
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

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


def read_pdf_with_unstructured(
    pdf_path: str,
    strategy: str = "auto",
    include_page_breaks: bool = False,
    infer_table_structure: bool = True
) -> Optional[List]:
    """
    Read and parse PDF file using Unstructured library
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Partitioning strategy ("auto", "hi_res", "ocr_only", "fast")
        include_page_breaks: Whether to include page breaks in output
        infer_table_structure: Whether to infer table structure
        
    Returns:
        List of document elements, or None if error
    """
    if not UNSTRUCTURED_AVAILABLE:
        raise ImportError(
            "Unstructured library is not installed. "
            "Please install it with: pip install 'unstructured[pdf]'"
        )
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Parsing PDF with Unstructured: {pdf_path}")
    print(f"Using strategy: {strategy}")
    
    try:
        # Method 1: Using partition_pdf (recommended for PDFs)
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            include_page_breaks=include_page_breaks,
            infer_table_structure=infer_table_structure
        )
        
        print(f"Successfully extracted {len(elements)} elements from PDF")
        return elements
        
    except Exception as e:
        print(f"Error parsing PDF with partition_pdf: {e}")
        print("Trying alternative method with partition.auto...")
        
        try:
            # Method 2: Using partition.auto (fallback)
            elements = partition(
                filename=pdf_path,
                strategy=strategy
            )
            print(f"Successfully extracted {len(elements)} elements using partition.auto")
            return elements
        except Exception as e2:
            print(f"Error with partition.auto: {e2}")
            return None


def extract_text_from_elements(elements: List) -> str:
    """
    Extract text content from document elements
    
    Args:
        elements: List of document elements
        
    Returns:
        Combined text content
    """
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
    """
    Analyze and categorize document elements
    
    Args:
        elements: List of document elements
        
    Returns:
        Dictionary with element type statistics
    """
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
