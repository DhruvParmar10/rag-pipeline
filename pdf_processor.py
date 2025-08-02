import asyncio
import hashlib
import io
import re
from typing import List, Tuple, Optional
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from models import DocumentChunk

settings = get_settings()

class PDFProcessor:
    """Advanced PDF processing with OCR fallback for scanned documents"""
    
    def __init__(self):
        self.max_file_size = settings.max_pdf_size_mb * 1024 * 1024
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_pdf(self, url: str) -> bytes:
        """Download PDF with retry logic and size validation"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(f"PDF file too large: {content_length} bytes")
                
                content = response.content
                if len(content) > self.max_file_size:
                    raise ValueError(f"PDF file too large: {len(content)} bytes")
                
                # Validate PDF header
                if not content.startswith(b'%PDF'):
                    raise ValueError("Invalid PDF file format")
                
                return content
                
            except httpx.RequestError as e:
                logger.error(f"Error downloading PDF from {url}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error downloading PDF: {e}")
                raise

    def extract_text_with_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber (better for structured PDFs)"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return ""

    def extract_text_with_pymupdf(self, pdf_content: bytes) -> str:
        """Extract text using PyMuPDF (fallback method)"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return ""

    def extract_text_with_ocr(self, pdf_content: bytes) -> str:
        """Extract text using OCR for scanned PDFs"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text_parts = []
            
            for page_num in range(min(doc.page_count, 10)):  # Limit OCR to first 10 pages
                page = doc[page_num]
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(img, lang='eng')
                if ocr_text.strip():
                    text_parts.append(ocr_text)
            
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""

    async def extract_text(self, pdf_content: bytes) -> str:
        """Extract text using multiple methods with fallbacks"""
        logger.info("Starting PDF text extraction")
        
        # Method 1: Try pdfplumber first (best for structured PDFs)
        text = self.extract_text_with_pdfplumber(pdf_content)
        if self._is_text_valid(text):
            logger.info("Text extracted successfully using pdfplumber")
            return self._clean_text(text)
        
        # Method 2: Try PyMuPDF as fallback
        text = self.extract_text_with_pymupdf(pdf_content)
        if self._is_text_valid(text):
            logger.info("Text extracted successfully using PyMuPDF")
            return self._clean_text(text)
        
        # Method 3: Try OCR for scanned PDFs
        logger.info("Attempting OCR extraction for scanned PDF")
        text = self.extract_text_with_ocr(pdf_content)
        if self._is_text_valid(text):
            logger.info("Text extracted successfully using OCR")
            return self._clean_text(text)
        
        raise ValueError("Failed to extract readable text from PDF using all available methods")

    def _is_text_valid(self, text: str) -> bool:
        """Check if extracted text is valid and meaningful"""
        if not text or len(text.strip()) < 100:
            return False
        
        # Check for reasonable character distribution
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        total_chars = len(text)
        
        if total_chars == 0:
            return False
            
        alphanumeric_ratio = alphanumeric_chars / total_chars
        return alphanumeric_ratio > 0.1  # At least 10% alphanumeric characters

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null characters
        text = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)  # Control characters
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

class TextChunker:
    """Intelligent text chunking with semantic awareness"""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.overlap_ratio = settings.chunk_overlap
        self.overlap_size = int(self.chunk_size * self.overlap_ratio)

    def create_chunks(self, text: str, document_url: str) -> List[DocumentChunk]:
        """Create overlapping chunks with metadata"""
        if not text.strip():
            raise ValueError("Cannot create chunks from empty text")
        
        # Split into sentences for better semantic boundaries
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_id = self._generate_chunk_id(document_url, chunk_index)
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        "document_url": document_url,
                        "total_chars": len(current_chunk),
                        "word_count": len(current_chunk.split())
                    }
                )
                chunks.append(chunk)
                
                # Prepare next chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text)
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk if there's remaining text
        if current_chunk.strip():
            chunk_id = self._generate_chunk_id(document_url, chunk_index)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={
                    "document_url": document_url,
                    "total_chars": len(current_chunk),
                    "word_count": len(current_chunk.split())
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns"""
        # Enhanced sentence splitting pattern
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out very short sentences and clean up
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = chunk.split()
        overlap_words = int(len(words) * self.overlap_ratio)
        overlap_words = max(1, min(overlap_words, len(words) // 2))
        return " ".join(words[-overlap_words:])

    def _generate_chunk_id(self, document_url: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        return f"chunk_{url_hash}_{chunk_index:04d}"
