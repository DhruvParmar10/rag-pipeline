import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore',
        protected_namespaces=()
    )
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_timeout: int = 120
    
    # Security
    bearer_token: str = "7c695e780a6ab6eacffab7c9326e5d8e472a634870a6365979c5671ad28f003c"
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generative_model: str = "qwen/qwen-2.5-72b-instruct"
    openrouter_api_key: Optional[str] = None
    use_openrouter: bool = True
    max_sequence_length: int = 4096
    embedding_dimension: int = 384
    
    # Qdrant Configuration
    qdrant_host: str = "https://e3684ffe-4ac5-439b-9f91-6c43f5af9e78.eu-central-1-0.aws.cloud.qdrant.io:6333"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "pdf_documents"
    qdrant_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Document Processing - Optimized for better chunking
    chunk_size: int = 800  # Increased from 500 for more context per chunk
    chunk_overlap: float = 0.3  # Increased from 0.25 for better continuity
    max_pdf_size_mb: int = 50
    max_questions: int = 20
    
    # Retrieval Configuration - Optimized for better search
    top_k_chunks: int = 10  # Increased from 5 for more comprehensive retrieval
    similarity_threshold: float = 0.5  # Reduced from 0.7 for more inclusive retrieval
    context_window: int = 4000  # Increased from 2000 for richer context
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
