import pytest
from src.loader import DocumentLoader


def test_load_pdf(tmp_path):
    # Create a dummy PDF (or skip if no file)
    # For now, we just check if function exists
    assert hasattr(DocumentLoader, "load_pdf")
