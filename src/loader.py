"""
Document Loader Module
Handles loading and extracting text from PDF and TXT files.
"""

import PyPDF2
import os
from typing import Optional


class DocumentLoader:
    """A safe document loader for PDF and plain text files."""

    @staticmethod
    def load_pdf(file_path: str, max_pages: Optional[int] = None) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.
            max_pages: Limit pages to load (useful for large docs).

        Returns:
            Extracted text as a single string.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        text_parts = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)

            pages_to_read = min(total_pages, max_pages) if max_pages else total_pages

            for i in range(pages_to_read):
                page = reader.pages[i]
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted.strip())

        return "\n\n".join(text_parts)

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load raw text from a .txt file with UTF-8 encoding."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
