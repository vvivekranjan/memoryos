import asyncio
import logging
import pymupdf

from aimemoryos.core.exceptions import IngestionError

logger = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    r"(?i)confidential information",
    r"(?i)all rights reserved",
    r"(?i)page \d+ of \d+",
]

class PDFLoader:
    def __init__(self):
        try:
            self.pymupdf = pymupdf
        except ImportError:
            logger.warning("PyMuPDF is not installed. PDF ingestion will fail.")
            self.pymupdf = None
            
    async def extract_text(self, filepath: str) -> str:
        if self.pymupdf is None:
            raise IngestionError("PyMuPDF is required for PDF ingestion. Run `pip install PyMuPDF`.")
            
        def _extract():
            try:
                doc = self.pymupdf.open(filepath)
                text_blocks = []
                for page in doc:
                    text_blocks.append(page.get_text())
                raw_text = "\n\n".join(text_blocks)
                return _strip_boilerplate(raw_text)
            except Exception as e:
                raise IngestionError(f"Failed to read PDF {filepath}: {e}")
                
        return await asyncio.to_thread(_extract)
    
    def _strip_boilerplate(self, text: str) -> str:
        for pattern in BOILERPLATE_PATTERNS:
            text = re.sub(pattern, " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)  # Replace multiple newlines with a single newline
        return text.strip()

