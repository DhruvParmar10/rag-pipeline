import asyncio
import hashlib
import re
from typing import List, Optional, Dict, Any
import numpy as np
import httpx
import json
from loguru import logger
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

# Gemini embedding service
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingService:
    """Service for generating embeddings using Google Gemini API"""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required in .env")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model_name = "models/text-embedding-004"  # 768 dimensions
        logger.info(f"Initialized Gemini embedding service with model: {self.model_name}")

    async def generate_embedding(self, text: str) -> list:
        """Generate embedding for a single text using Gemini API with caching"""
        try:
            # Create a cache key based on text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Use Gemini's embedding model
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
                
        except Exception as e:
            logger.error(f"Embedding generation failed for text length {len(text)}: {e}")
            raise

    async def generate_batch_embeddings(self, texts: list) -> list:
        """Generate embeddings for a batch of texts focusing on quality"""
        try:
            # Step 1: Remove duplicates to avoid redundant processing
            unique_texts = []
            text_to_index = {}
            original_indices = []
            
            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash not in text_to_index:
                    text_to_index[text_hash] = len(unique_texts)
                    unique_texts.append(text)
                original_indices.append(text_to_index[text_hash])
            
            if len(unique_texts) < len(texts):
                logger.info(f"Deduplication: Processing {len(unique_texts)} unique texts from {len(texts)} total")
            
            # Step 2: Process unique texts with quality-focused batching
            batch_size = 25  # Smaller batches for better quality and reliability
            max_concurrent = 10  # Conservative concurrency for stability
            unique_embeddings = []
            
            # Create semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(max_concurrent)
            
            for i in range(0, len(unique_texts), batch_size):
                batch = unique_texts[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(unique_texts) + batch_size - 1)//batch_size
                progress = (batch_num / total_batches) * 100
                logger.info(f"Processing embedding batch {batch_num}/{total_batches} ({progress:.1f}% complete) - {len(batch)} texts")
                
                # Process the entire batch concurrently with quality controls
                async def process_text_with_semaphore(text):
                    async with semaphore:
                        return await self._generate_single_embedding_quality(text)
                
                # Create all tasks for the batch
                tasks = [process_text_with_semaphore(text) for text in batch]
                
                # Execute all tasks in parallel with generous timeout
                try:
                    batch_embeddings = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=120.0  # Generous timeout for quality
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Batch {batch_num} timed out, retrying with single requests")
                    # Fallback to individual processing
                    batch_embeddings = []
                    for text in batch:
                        try:
                            embedding = await self._generate_single_embedding_quality(text)
                            batch_embeddings.append(embedding)
                        except Exception as e:
                            logger.warning(f"Failed individual embedding: {e}")
                            batch_embeddings.append([0.0] * 768)
                
                # Handle any exceptions and filter successful results
                successful_embeddings = []
                for j, result in enumerate(batch_embeddings):
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to generate embedding for text {i+j}: {result}")
                        # Retry once for failed embeddings
                        try:
                            retry_embedding = await self._generate_single_embedding_quality(batch[j])
                            successful_embeddings.append(retry_embedding)
                        except Exception:
                            # Use a zero vector as final fallback
                            successful_embeddings.append([0.0] * 768)
                    else:
                        successful_embeddings.append(result)
                
                unique_embeddings.extend(successful_embeddings)
                
                # Small delay between batches for API stability
                if i + batch_size < len(unique_texts):
                    await asyncio.sleep(0.5)
            
            # Step 3: Map back to original order with duplicates
            all_embeddings = [unique_embeddings[idx] for idx in original_indices]
                
            logger.info(f"Generated {len(all_embeddings)} embeddings with quality focus")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise

    async def _generate_single_embedding_quality(self, text: str) -> list:
        """Generate a single embedding with quality focus"""
        try:
            # Quality-focused text preprocessing
            processed_text = self._preprocess_text_for_quality(text)
            
            # Use Gemini API with quality settings
            result = genai.embed_content(
                model=self.model_name,
                content=processed_text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Quality embedding failed for text length {len(text)}: {e}")
            # Retry with shorter text if failed
            try:
                short_text = text[:1000] if len(text) > 1000 else text
                result = genai.embed_content(
                    model=self.model_name,
                    content=short_text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                raise

    def _preprocess_text_for_quality(self, text: str, max_length: int = 2000) -> str:
        """Preprocess text for quality embeddings"""
        # Clean up the text
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # If text is too long, intelligently truncate
        if len(text) <= max_length:
            return text
        
        # Try to keep complete sentences
        sentences = text.split('.')
        truncated_text = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(truncated_text + sentence + '.') <= max_length:
                truncated_text += sentence + '. '
            else:
                break
        
        # If no complete sentences fit, just truncate
        if not truncated_text.strip():
            truncated_text = text[:max_length-3] + "..."
        
        return truncated_text.strip()


class GenerationService:
    """Service for generating text completions using OpenRouter API"""
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = os.getenv("GENERATIVE_MODEL", "qwen/qwen-2.5-72b-instruct")  
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
            "Answer:", "Response:", "Based on the document context provided below,",
            "Document Context:", "Question:"
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
                    "content": "You are a helpful AI assistant that answers questions based on document context. Extract relevant information directly from the provided context to answer questions concisely."
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
        prompt = f"""Based on the document context provided below, answer the question directly and concisely.

Document Context:
{context}

Question: {question}

Answer: """
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
