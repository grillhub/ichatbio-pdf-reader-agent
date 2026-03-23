"""
PDF reading utilities using pypdf library
Extracts PDF URLs from text, downloads PDFs, and extracts text content
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any

# Try to import pypdf
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("Warning: pypdf library not installed. Install with: pip install pypdf")


def extract_pdf_urls_from_text(text: str) -> List[str]:
    """
    Extract PDF URLs from text using regex patterns
    
    Args:
        text: Text to search for PDF URLs
        
    Returns:
        List of PDF URLs found in the text
    """
    # Pattern to match URLs ending in .pdf or containing /pdf/ in the path
    # Also matches common PDF hosting patterns like arxiv.org/pdf/, etc.
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
    """
    Download PDF from URL
    
    Args:
        url: URL of the PDF file
        output_path: Local path to save the PDF
        
    Returns:
        Path to the downloaded PDF file
    """
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


def read_pdf_with_pypdf(
    pdf_path: str,
    strategy: str = "auto",
    include_page_breaks: bool = False,
    infer_table_structure: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Read and parse PDF file using pypdf library
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Strategy parameter (kept for compatibility, not used with pypdf)
        include_page_breaks: Whether to include page breaks in output
        infer_table_structure: Whether to infer table structure (kept for compatibility, not used with pypdf)
        
    Returns:
        List of document elements (simulated structure for compatibility), or None if error
    """
    if not PYPDF_AVAILABLE:
        raise ImportError(
            "pypdf library is not installed. "
            "Please install it with: pip install pypdf"
        )
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Parsing PDF: {pdf_path}")
    
    try:
        # Read PDF with pypdf
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF has {num_pages} pages")
        
        # Extract text from each page
        elements = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    # Create a simple element structure for compatibility
                    element = {
                        'type': 'Text',
                        'text': text.strip(),
                        'page_number': page_num,
                        'metadata': {
                            'page_number': page_num,
                            'total_pages': num_pages
                        }
                    }
                    elements.append(element)
                    
                    # Add page break if requested
                    if include_page_breaks and page_num < num_pages:
                        elements.append({
                            'type': 'PageBreak',
                            'text': f'\n--- Page {page_num + 1} ---\n',
                            'page_number': page_num + 1,
                            'metadata': {'page_number': page_num + 1}
                        })
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page_num}: {e}")
                continue
        
        print(f"Successfully extracted {len(elements)} text elements from PDF")
        return elements
        
    except Exception as e:
        print(f"Error parsing PDF with pypdf: {e}")
        return None


# Alias for backward compatibility
def read_pdf_with_unstructured(
    pdf_path: str,
    strategy: str = "auto",
    include_page_breaks: bool = False,
    infer_table_structure: bool = True
) -> Optional[List]:
    """
    Alias for read_pdf_with_pypdf (for backward compatibility)
    """
    return read_pdf_with_pypdf(pdf_path, strategy, include_page_breaks, infer_table_structure)


def extract_text_from_elements(elements: List) -> str:
    """
    Extract text content from document elements
    
    Args:
        elements: List of document elements (can be dicts from pypdf or unstructured objects)
        
    Returns:
        Combined text content
    """
    text_parts = []
    for element in elements:
        # Handle dict elements (from pypdf)
        if isinstance(element, dict):
            text = element.get('text', '')
        # Handle object elements (from unstructured)
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
    """
    Analyze and categorize document elements
    
    Args:
        elements: List of document elements (can be dicts from pypdf or unstructured objects)
        
    Returns:
        Dictionary with element type statistics
    """
    element_stats = {}
    element_types = []
    
    for element in elements:
        # Handle dict elements (from pypdf)
        if isinstance(element, dict):
            element_type = element.get('type', 'Unknown')
        # Handle object elements (from unstructured)
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
    """
    Save extracted content to file
    
    Args:
        content: Text content to save
        output_dir: Output directory
        filename: Output filename
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Content saved to {output_path}")
    return output_path


def save_elements_json(elements: List, output_dir: str, filename: str = "elements.json"):
    """
    Save elements as JSON (if supported)
    
    Args:
        elements: List of document elements
        output_dir: Output directory
        filename: Output filename
    """
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


def main():
    """Main function to download and read PDF from arXiv"""
    # Configuration
    # pdf_url = "https://arxiv.org/pdf/2408.09869"
    # pdf_filename = "2408.09869.pdf"
    pdf_url = "https://arxiv.org/pdf/2408.09869"
    pdf_filename = "test.pdf"
    output_dir = "./output_unstructured"
    
    try:
        # Download PDF
        print("="*80)
        print("PDF Processing with Unstructured Library")
        print("="*80)
        download_pdf(pdf_url, pdf_filename)
        
        # Read PDF with pypdf
        elements = read_pdf_with_pypdf(
            pdf_path=pdf_filename,
            strategy="auto",
            include_page_breaks=False,
            infer_table_structure=True
        )
        
        if not elements:
            print("Failed to extract elements from PDF")
            return
        
        # Analyze elements
        stats = analyze_elements(elements)
        print("\n" + "="*80)
        print("Element Analysis:")
        print("="*80)
        print(f"Total elements extracted: {stats['total_elements']}")
        print("\nElement types found:")
        for el_type, count in sorted(stats['element_types'].items()):
            print(f"  - {el_type}: {count}")
        
        # Extract text content
        print("\n" + "="*80)
        print("Extracting text content...")
        print("="*80)
        text_content = extract_text_from_elements(elements)
        
        # Display preview
        preview_length = 2000
        print("\n" + "="*80)
        print(f"Preview of extracted content (first {preview_length} characters):")
        print("="*80)
        print(text_content[:preview_length])
        print("\n... (content truncated)")
        print(f"\nTotal content length: {len(text_content)} characters")
        
        # Save outputs
        print("\n" + "="*80)
        print("Saving outputs...")
        print("="*80)
        save_content(text_content, output_dir, "extracted_text.txt")
        save_elements_json(elements, output_dir, "elements.json")
        
        print("\n" + "="*80)
        print("Processing complete!")
        print("="*80)
        print(f"Output files saved to: {output_dir}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nInstallation instructions:")
        print("  pip install pypdf")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
