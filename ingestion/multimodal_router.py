import os
import asyncio
from typing import Optional
from core.exceptions import IngestionError

class MultimodalRouter:
    def __init__(self, pdf_loader):
        self.pdf_loader = pdf_loader
        
    async def route_and_extract(self, source: str) -> str:
        """Route the source based on file extension or content type."""
        # Check if it's a file path
        if os.path.exists(source):
            ext = os.path.splitext(source)[1].lower()
            if ext == '.pdf':
                return await self.pdf_loader.extract_text(source)
            elif ext in ('.txt', '.md', '.json', '.csv'):
                def _read():
                    with open(source, 'r', encoding='utf-8') as f:
                        return f.read()
                return await asyncio.to_thread(_read)
            else:
                raise IngestionError(f"Unsupported file type: {ext}")
        
        # If it's not a file path, assume it's raw text
        return str(source)
