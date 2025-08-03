import asyncio
import hashlib
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import httpx
import json
from loguru import logger
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from models import SearchResult

settings = get_settings()

class EmbeddingService:
    """Service for generating embeddings using sentence transformers"""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.redis_client = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize embedding model and Redis cache"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded on {self.device}")
            
            # Initialize Redis for caching
            if settings.redis_host:
                try:
                    self.redis_client = redis.Redis(
                        host=settings.redis_host,
                        port=settings.redis_port,
                        db=settings.redis_db,
                        password=settings.redis_password,
                        decode_responses=True,
                        socket_connect_timeout=2,
                        socket_timeout=2
                    )
                    # Test connection
                    self.redis_client.ping()
                    logger.info("Redis cache initialized and connected")
                except Exception as e:
                    logger.warning(f"Redis connection failed, disabling cache: {e}")
                    self.redis_client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching"""
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if self.redis_client:
            cached_embedding = await self._get_cached_embedding(cache_key)
            if cached_embedding:
                return cached_embedding
        
        # Generate embedding
        try:
            # Run embedding generation in thread pool for async compatibility
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self._generate_embedding_sync, 
                text
            )
            
            # Cache the result
            if self.redis_client:
                await self._cache_embedding(cache_key, embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _generate_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding generation"""
        embedding = self.model.encode([text], convert_to_tensor=False)[0]
        return embedding

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided for embedding generation")
        
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._generate_batch_embeddings_sync,
                valid_texts
            )
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def _generate_batch_embeddings_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous batch embedding generation"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embedding:{self.model_name}:{text_hash}"

    async def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        try:
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        return None

    async def _cache_embedding(self, cache_key: str, embedding: List[float], ttl: int = 86400):
        """Cache embedding with TTL (default 24 hours)"""
        try:
            if self.redis_client:
                # Convert numpy array to list if needed
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                self.redis_client.setex(cache_key, ttl, json.dumps(embedding))
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

class GenerativeModel:
    """Service for text generation using OpenRouter API with Qwen models"""
    
    def __init__(self):
        self.model_name = settings.generative_model
        self.api_key = settings.openrouter_api_key
        self.use_openrouter = settings.use_openrouter
        self.max_length = settings.max_sequence_length
        self.base_url = "https://openrouter.ai/api/v1"
        
        if self.use_openrouter and not self.api_key:
            raise ValueError("OpenRouter API key is required when use_openrouter is True")
        
        logger.info(f"Initialized GenerativeModel with {'OpenRouter API' if self.use_openrouter else 'local model'}")

    async def generate_enhanced_answer(self, question: str, context: str) -> str:
        """Generate enhanced answer with better prompting and handling"""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        if not context.strip():
            return "The document does not contain information to answer this question."
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(question, context)
        
        try:
            if self.use_openrouter:
                answer = await self._generate_with_openrouter_enhanced(prompt)
            else:
                answer = await self._generate_with_local_model(prompt)
            
            # Clean and validate answer
            cleaned_answer = self._clean_and_validate_answer(answer, question)
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."

    def _create_enhanced_prompt(self, question: str, context: str) -> str:
        """Create an enhanced prompt for better answer generation"""
        prompt = f"""You are an expert document analyst. Your task is to answer questions based STRICTLY on the provided context from a document.

INSTRUCTIONS:
1. Read the context carefully and identify relevant information
2. Answer the question using ONLY information explicitly stated in the context
3. If specific details are mentioned, include them in your answer
4. Quote exact phrases when they directly answer the question
5. Keep answers concise but complete (2-4 sentences maximum)
6. If the context doesn't contain the answer, respond: "The document does not contain information to answer this question."

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

Based on the context provided above, provide a precise and factual answer:"""
        return prompt

    async def _generate_with_openrouter_enhanced(self, prompt: str) -> str:
        """Enhanced generation using OpenRouter API with optimized parameters"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/rag-pipeline",
            "X-Title": "RAG PDF QA Pipeline"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise document analyst that provides accurate, concise answers based strictly on document context. Never add external information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,  # Increased slightly for more complete answers
            "temperature": 0.1,  # Very low for factual responses
            "top_p": 0.9,
            "frequency_penalty": 0.1,  # Slight penalty to avoid repetition
            "presence_penalty": 0.1,   # Encourage focused responses
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("No response generated from OpenRouter API")

    def _clean_and_validate_answer(self, answer: str, question: str) -> str:
        """Clean and validate the generated answer"""
        if not answer or not answer.strip():
            return "The document does not contain information to answer this question."
        
        # Clean the answer
        cleaned = answer.strip()
        
        # Remove any unwanted prefixes that the model might add
        unwanted_prefixes = [
            "Based on the context,", "According to the document,", 
            "The document states that", "From the context provided,",
            "Answer:", "Response:"
        ]
        
        for prefix in unwanted_prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Ensure the answer isn't just repeating the question
        if cleaned.lower() == question.lower():
            return "The document does not contain information to answer this question."
        
        # Ensure reasonable length
        if len(cleaned) < 10:
            return "The document does not contain sufficient information to answer this question."
        
        # Limit length if too long
        if len(cleaned) > 800:
            sentences = cleaned.split('. ')
            truncated = '. '.join(sentences[:3])
            if not truncated.endswith('.'):
                truncated += '.'
            cleaned = truncated
        
        return cleaned

    async def _generate_with_openrouter(self, prompt: str) -> str:
        """Generate answer using OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/rag-pipeline",
            "X-Title": "RAG PDF QA Pipeline"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise document analyzer. Provide short, accurate answers based strictly on the given context. Never add information not explicitly stated in the document."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 200,  # Reduced for shorter answers
            "temperature": 0.3,  # Lower temperature for more focused responses
            "top_p": 0.9,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("No response generated from OpenRouter API")

    async def _generate_with_local_model(self, prompt: str) -> str:
        """Fallback method for local model generation (not implemented)"""
        raise NotImplementedError("Local model generation not implemented. Please use OpenRouter API.")

    def _create_prompt(self, question: str, context: str) -> str:
        """Create a well-structured prompt for the model"""
        prompt = f"""You are an AI assistant that answers questions strictly based on the provided document context. Follow these rules:

1. Only use information explicitly stated in the context
2. Keep answers short and concise (1-3 sentences maximum)
3. Do not add any external knowledge or assumptions
4. If the exact answer is not in the context, say "The document does not specify this information"
5. Quote specific phrases from the document when possible

Context:
{context}

Question: {question}

Provide a brief, direct answer based only on the document context:"""
        return prompt

    def _clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer"""
        # Remove the original prompt if it's still there
        if "Context:" in answer:
            answer = answer.split("Context:")[-1]
        if "Question:" in answer:
            answer = answer.split("Question:")[-1]
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1]
        if "Provide a brief" in answer:
            # Remove any instruction remnants
            lines = answer.split('\n')
            for i, line in enumerate(lines):
                if not any(keyword in line.lower() for keyword in ['provide', 'brief', 'direct', 'based only']):
                    answer = '\n'.join(lines[i:])
                    break
        
        # Clean up formatting
        answer = answer.strip()
        
        # Remove excessive whitespace
        answer = " ".join(answer.split())
        
        # Ensure answer is concise (max 300 characters for short answers)
        if len(answer) > 300:
            sentences = answer.split('.')
            truncated_sentences = []
            char_count = 0
            
            for sentence in sentences:
                if char_count + len(sentence) > 250:  # Reduced limit
                    break
                truncated_sentences.append(sentence)
                char_count += len(sentence)
            
            answer = '.'.join(truncated_sentences)
            if not answer.endswith('.'):
                answer += '.'
        
        return answer if answer else "The document does not contain information to answer this question."
