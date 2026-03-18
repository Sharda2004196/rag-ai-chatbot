"""
Document Parser - Extracts text from various file formats
Supports: PDF, DOCX, TXT, MD
"""

import os
from typing import Optional

def parse_document(file_path: str) -> Optional[str]:
    """
    Parse a document and extract text content

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text content or None if parsing fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.pdf':
            return parse_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return parse_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            return parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        raise Exception(f"Failed to parse {file_ext} file: {str(e)}")


def parse_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        text = []

        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text.append(f"[Page {page_num}]\n{page_text}")

        if not text:
            raise Exception("No text could be extracted from PDF")

        return "\n\n".join(text)

    except ImportError:
        raise Exception("PyPDF2 not installed. Run: pip install PyPDF2")


def parse_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document

        doc = Document(file_path)
        text = []

        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)

        if not text:
            raise Exception("No text could be extracted from DOCX")

        return "\n\n".join(text)

    except ImportError:
        raise Exception("python-docx not installed. Run: pip install python-docx")


def parse_text(file_path: str) -> str:
    """Extract text from plain text file"""
    encodings = ['utf-8', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise Exception("Could not decode text file with any supported encoding")


def parse_file_content(content: bytes, filename: str) -> str:
    """
    Parse file content from bytes

    Args:
        content: File content as bytes
        filename: Original filename (used to determine file type)

    Returns:
        Extracted text content
    """
    import tempfile

    # Save to temporary file
    file_ext = os.path.splitext(filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        text = parse_document(tmp_path)
        return text
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
