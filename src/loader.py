import PyPDF2
import os

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load raw text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()