import os
import asyncio
from typing import Optional
from aimemoryos.core.exceptions import IngestionError

class MultimodalRouter:
    def __init__(self, pdf_loader):
        self.pdf_loader = pdf_loader
        
    async def route_and_extract(self, source: str) -> str:
        """Route the source based on file extension or content type."""
        # Check if it's a file path
        is_path = False
        try:
            if len(source) < 2048 and os.path.exists(source):
                is_path = True
        except Exception:
            pass

        if is_path:
            abs_source = os.path.abspath(source)
            trusted_root = os.path.abspath(os.getcwd())
            if not abs_source.startswith(trusted_root):
                raise IngestionError("Path traversal detected: source is outside trusted root")
                
            if os.path.getsize(abs_source) > 50 * 1024 * 1024:
                raise IngestionError("File size exceeds 50MB limit")

            ext = os.path.splitext(abs_source)[1].lower()
            if ext == '.pdf':
                return await self.pdf_loader.extract_text(abs_source)
            elif ext in ('.txt', '.md', '.json', '.csv'):
                def _read():
                    with open(abs_source, 'r', encoding='utf-8') as f:
                        return f.read()
                return await asyncio.to_thread(_read)
            else:
                raise IngestionError(f"Unsupported file type: {ext}")
        
        # If it's not a file path, assume it's raw text
        return str(source)
