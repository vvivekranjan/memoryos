import asyncio
import logging
import pymupdf

from aimemoryos.core.exceptions import IngestionError

logger = logging.getLogger(__name__)

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
                return "\n\n".join(text_blocks)
            except Exception as e:
                raise IngestionError(f"Failed to read PDF {filepath}: {e}")
                
        return await asyncio.to_thread(_extract)