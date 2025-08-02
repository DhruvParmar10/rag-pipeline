from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    SCANNED_PDF = "scanned_pdf"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class HackRXRequest(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    documents: HttpUrl
    questions: List[str] = Field(..., min_items=1, max_items=20)

class HackRXResponse(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    answers: List[str]

class DocumentChunk(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    chunk_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = {}

class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    text: str
    
class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    embedding: List[float]
    model_used: str

class SearchResult(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    model_config = ConfigDict(
        extra='ignore',
        protected_namespaces=()
    )
    
    status: str
    services: Dict[str, str]
    timestamp: str
