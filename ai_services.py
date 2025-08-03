import asyncio
import hashlib
from typing import List, Optional, Dict, Any
import numpy as np
import httpx
import json
from loguru import logger
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

# Free embedding service using Hugging Face Inference API
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingService:
    """Service for generating embeddings using free Hugging Face Inference API"""
    def __init__(self):
        self.api_key = os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN (Hugging Face API key) is required in .env")
        
        # Use standard HF Inference API (free tier) without provider
        self.client = InferenceClient(token=self.api_key)
        # Use a free, efficient model that's supported by standard HF API
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions, free
        logger.info(f"Initialized free embedding service with model: {self.model_id}")

    async def generate_embedding(self, text: str) -> list:
        """Generate embedding for a single text using free Hugging Face Inference API"""
        try:
            # Use the standard feature_extraction method (free tier)
            result = self.client.feature_extraction(text, model=self.model_id)
            
            # Handle different response formats
            if hasattr(result, 'tolist'):  # numpy array
                if result.ndim == 2 and result.shape[0] == 1:
                    return result[0].tolist()
                elif result.ndim == 1:
                    return result.tolist()
                return result.flatten().tolist()
            elif isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    return result[0]  # Extract inner vector
                return result
            else:
                # Convert to list if it's another format
                return list(result) if hasattr(result, '__iter__') else [result]
                
        except Exception as e:
            # Log the error but try to provide more context
            logger.error(f"Embedding generation failed for text length {len(text)}: {e}")
            if "402" in str(e) or "payment" in str(e).lower():
                raise ValueError("Embedding service requires payment. Please use a different model or provider.")
            raise

    async def generate_batch_embeddings(self, texts: list) -> list:
        """Generate embeddings for a batch of texts using free Hugging Face Inference API"""
        try:
            # Process in smaller batches to avoid timeouts and improve efficiency
            batch_size = 5  # Process 5 texts at a time
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Process batch concurrently
                batch_embeddings = []
                for text in batch:
                    embedding = await self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
                
            logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise


class GenerationService:
    """Service for generating text completions using OpenRouter API"""
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("GENERATION_MODEL", "qwen/qwen3-8b")  
        self.base_url = "https://openrouter.ai/api/v1"
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required in .env")

    async def generate(self, question: str, context: str) -> str:
        """Generate answer based on question and context"""
        prompt = self._create_prompt(question, context)
        try:
            raw_answer = await self._generate_with_openrouter(prompt)
            cleaned_answer = self._clean_and_validate_answer(raw_answer, question)
            return cleaned_answer
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, but I'm unable to generate an answer at the moment. Please try again."

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
