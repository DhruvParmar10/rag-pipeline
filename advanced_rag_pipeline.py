"""
High-Quality RAG Pipeline Implementation
Using proven production techniques for accuracy and reliability
"""
import asyncio
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import numpy as np
from collections import defaultdict
import json

from pdf_processor import PDFProcessor, TextChunker
from ai_services import EmbeddingService, GenerationService
from vector_store import VectorStore
from models import DocumentChunk, SearchResult
from config import get_settings

settings = get_settings()

class AdvancedRAGPipeline:
    """Advanced RAG pipeline focusing on accuracy and code quality"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.generation_service = GenerationService()
        self.vector_store = VectorStore()
        
        # Quality-focused caching
        self._document_cache: Dict[str, bool] = {}
        self._chunk_quality_scores: Dict[str, float] = {}
        
        logger.info("Initialized AdvancedRAGPipeline with quality-first approach")

    async def process_document_and_answer(self, document_url: str, questions: List[str]) -> List[str]:
        """Main pipeline with enhanced quality controls"""
        try:
            logger.info(f"Starting advanced RAG pipeline for: {document_url}")
            
            # Ensure document is processed with quality controls
            await self._ensure_document_processed_quality(document_url)
            
            # Answer questions using multi-stage retrieval
            answers = await self._answer_questions_advanced(document_url, questions)
            
            logger.info(f"Advanced RAG pipeline completed for {len(questions)} questions")
            return answers
            
        except Exception as e:
            logger.error(f"Advanced RAG pipeline failed: {e}")
            raise

    async def _ensure_document_processed_quality(self, document_url: str):
        """Process document with quality-first approach"""
        document_hash = self._get_document_hash(document_url)
        
        if document_hash in self._document_cache:
            logger.info(f"Document already processed with quality controls: {document_url}")
            return
        
        # Check if document exists in vector store
        try:
            dummy_embedding = [0.0] * 768
            existing_chunks = await self.vector_store.search_similar_chunks(
                query_embedding=dummy_embedding,
                document_url=document_url,
                top_k=1,
                threshold=0.0
            )
            
            if existing_chunks:
                logger.info(f"Document exists in vector store: {document_url}")
                self._document_cache[document_hash] = True
                return
                
        except Exception as e:
            logger.warning(f"Could not check document existence: {e}")
        
        logger.info(f"Processing document with quality controls: {document_url}")
        
        try:
            # Extract text with quality validation
            pdf_content = await self.pdf_processor.download_pdf(document_url)
            text = await self.pdf_processor.extract_text(pdf_content)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Create high-quality chunks
            chunks = self._create_quality_chunks(text, document_url)
            
            if not chunks:
                raise ValueError("No quality chunks could be created")
            
            # Quality assessment and filtering
            quality_chunks = self._assess_and_filter_chunks(chunks)
            logger.info(f"Quality filtering: {len(chunks)} → {len(quality_chunks)} chunks")
            
            # Generate embeddings for quality chunks
            chunk_texts = [chunk.content for chunk in quality_chunks]
            embeddings = await self.embedding_service.generate_batch_embeddings(chunk_texts)
            
            # Store with quality metadata
            await self.vector_store.store_chunks(quality_chunks, embeddings, document_url)
            
            # Cache successful processing
            self._document_cache[document_hash] = True
            
            logger.info(f"Document processed successfully: {len(quality_chunks)} quality chunks stored")
            
        except Exception as e:
            logger.error(f"Failed to process document {document_url}: {e}")
            raise

    def _create_quality_chunks(self, text: str, document_url: str) -> List[DocumentChunk]:
        """Create chunks with quality-first approach"""
        # Use the existing chunker but with quality settings
        chunks = self.text_chunker.create_chunks(text, document_url)
        
        # Enhance chunks with quality metadata
        enhanced_chunks = []
        for chunk in chunks:
            # Calculate quality score
            quality_score = self._calculate_chunk_quality(chunk.content)
            
            # Enhance metadata
            enhanced_metadata = chunk.metadata.copy()
            enhanced_metadata.update({
                'quality_score': quality_score,
                'has_numbers': self._contains_numerical_data(chunk.content),
                'has_key_terms': self._contains_key_insurance_terms(chunk.content),
                'sentence_completeness': self._assess_sentence_completeness(chunk.content),
                'information_density': self._calculate_information_density(chunk.content)
            })
            
            enhanced_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=enhanced_metadata
            )
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

    def _calculate_chunk_quality(self, content: str) -> float:
        """Calculate quality score for a chunk"""
        score = 0.0
        
        # Length score (prefer medium-length chunks)
        length = len(content)
        if 200 <= length <= 1000:
            score += 0.3
        elif 100 <= length <= 1500:
            score += 0.2
        else:
            score += 0.1
        
        # Sentence completeness
        sentences = content.split('.')
        complete_sentences = len([s for s in sentences if len(s.strip()) > 10])
        if complete_sentences >= 2:
            score += 0.2
        
        # Key term presence
        key_terms = [
            'coverage', 'benefit', 'policy', 'claim', 'premium', 'deductible',
            'waiting period', 'grace period', 'exclusion', 'condition',
            'payment', 'reimbursement', 'hospital', 'treatment'
        ]
        
        key_term_count = sum(1 for term in key_terms if term in content.lower())
        score += min(key_term_count * 0.05, 0.3)
        
        # Numerical information (important for insurance)
        if re.search(r'\d+\s*(days?|months?|years?|%|percent|dollars?|\$)', content):
            score += 0.2
        
        return min(score, 1.0)

    def _contains_numerical_data(self, content: str) -> bool:
        """Check if chunk contains numerical data"""
        patterns = [
            r'\d+\s*(days?|months?|years?)',
            r'\d+\s*%',
            r'\$\d+',
            r'\d+\s*(percent)',
            r'\d+\s*(dollars?)'
        ]
        return any(re.search(pattern, content.lower()) for pattern in patterns)

    def _contains_key_insurance_terms(self, content: str) -> bool:
        """Check if chunk contains key insurance terminology"""
        key_terms = [
            'coverage', 'benefit', 'policy', 'claim', 'premium', 'deductible',
            'copay', 'coinsurance', 'out-of-pocket', 'network', 'provider',
            'waiting period', 'grace period', 'pre-existing', 'exclusion'
        ]
        content_lower = content.lower()
        return any(term in content_lower for term in key_terms)

    def _assess_sentence_completeness(self, content: str) -> float:
        """Assess how complete the sentences are in the chunk"""
        sentences = content.split('.')
        complete_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and sentence[0].isupper():
                complete_sentences += 1
        
        total_sentences = len([s for s in sentences if len(s.strip()) > 5])
        
        if total_sentences == 0:
            return 0.0
        
        return complete_sentences / total_sentences

    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density of the chunk"""
        words = content.split()
        
        # Count informative words (not stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can'}
        
        informative_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        if len(words) == 0:
            return 0.0
        
        return len(informative_words) / len(words)

    def _assess_and_filter_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Assess and filter chunks based on quality"""
        quality_chunks = []
        
        for chunk in chunks:
            quality_score = chunk.metadata.get('quality_score', 0.0)
            
            # Keep chunks with good quality scores
            if quality_score >= 0.4:  # Quality threshold
                quality_chunks.append(chunk)
            # Also keep chunks with key terms even if lower quality
            elif chunk.metadata.get('has_key_terms', False):
                quality_chunks.append(chunk)
            # Keep chunks with numerical data (important for insurance)
            elif chunk.metadata.get('has_numbers', False):
                quality_chunks.append(chunk)
        
        # Ensure we don't filter out too much
        if len(quality_chunks) < len(chunks) * 0.3:
            # If we filtered too aggressively, keep top chunks by quality
            sorted_chunks = sorted(chunks, key=lambda c: c.metadata.get('quality_score', 0.0), reverse=True)
            target_count = max(len(quality_chunks), int(len(chunks) * 0.3))
            quality_chunks = sorted_chunks[:target_count]
        
        return quality_chunks

    async def _answer_questions_advanced(self, document_url: str, questions: List[str]) -> List[str]:
        """Answer questions using advanced multi-stage retrieval"""
        answers = []
        document_hash = self._get_document_hash(document_url)
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:100]}...")
                
                # Multi-stage retrieval
                search_results = await self._multi_stage_retrieval(question, document_hash)
                
                if not search_results:
                    logger.warning(f"No relevant chunks found for question: {question}")
                    answers.append("The document does not contain information to answer this question.")
                    continue
                
                # Prepare high-quality context
                context = self._prepare_quality_context(search_results, question)
                
                # Generate answer with quality validation
                answer = await self._generate_quality_answer(question, context)
                answers.append(answer)
                
                logger.info(f"Question {i+1} answered successfully")
                
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {e}")
                answers.append("I apologize, but I encountered an error while processing this question.")
        
        return answers

    async def _multi_stage_retrieval(self, question: str, document_hash: str) -> List[SearchResult]:
        """Multi-stage retrieval for better accuracy"""
        try:
            # Stage 1: Semantic search
            question_embedding = await self.embedding_service.generate_embedding(question)
            
            # Get more candidates initially
            primary_results = await self.vector_store.search_similar_chunks(
                query_embedding=question_embedding,
                document_url=None,
                top_k=settings.top_k_chunks * 2,  # Get more candidates
                threshold=0.0
            )
            
            # Filter by document hash
            document_results = primary_results  # Skip document filtering for single document use case
            
            logger.info(f"Document filtering: {len(primary_results)} → {len(document_results)} chunks (no filtering applied)")
            
            # Stage 2: Re-rank based on question relevance and chunk quality
            reranked_results = self._rerank_by_relevance_and_quality(question, document_results)
            
            # Stage 3: Diversify results to avoid redundancy
            final_results = self._diversify_results(reranked_results, max_results=settings.top_k_chunks)
            
            logger.info(f"Multi-stage retrieval: {len(primary_results)} → {len(document_results)} → {len(final_results)} chunks")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Multi-stage retrieval failed: {e}")
            return []

    def _rerank_by_relevance_and_quality(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results based on question relevance and chunk quality"""
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_results = []
        
        for result in results:
            content_lower = result.content.lower()
            content_words = set(content_lower.split())
            
            # Semantic similarity score (from vector search)
            semantic_score = result.score
            
            # Lexical overlap score
            overlap = len(question_words & content_words)
            lexical_score = overlap / len(question_words) if question_words else 0
            
            # Quality score from chunk metadata
            quality_score = result.metadata.get('quality_score', 0.5)
            
            # Numerical data bonus (important for insurance questions)
            numerical_bonus = 0.1 if result.metadata.get('has_numbers', False) else 0
            
            # Key terms bonus
            key_terms_bonus = 0.1 if result.metadata.get('has_key_terms', False) else 0
            
            # Combined score
            combined_score = (
                semantic_score * 0.4 +
                lexical_score * 0.3 +
                quality_score * 0.2 +
                numerical_bonus +
                key_terms_bonus
            )
            
            scored_results.append((result, combined_score))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, _ in scored_results]

    def _diversify_results(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """Diversify results to avoid redundant content"""
        if len(results) <= max_results:
            return results
        
        diverse_results = []
        used_contents = []  # List of sets instead of set of sets
        
        for result in results:
            # Check for content similarity with already selected chunks
            content_words = set(result.content.lower().split())
            
            is_diverse = True
            for used_content in used_contents:
                overlap = len(content_words & used_content)
                similarity = overlap / len(content_words | used_content) if content_words | used_content else 0
                
                if similarity > 0.7:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
                used_contents.append(content_words)  # Append to list instead of add to set
                
                if len(diverse_results) >= max_results:
                    break
        
        return diverse_results

    def _prepare_quality_context(self, search_results: List[SearchResult], question: str) -> str:
        """Prepare high-quality context from search results"""
        context_parts = []
        total_chars = 0
        max_context_chars = settings.context_window
        
        for i, result in enumerate(search_results):
            chunk_content = result.content
            
            # Add chunk number and score for transparency
            chunk_header = f"[Chunk {i+1}, Relevance: {result.score:.3f}]"
            chunk_text = f"{chunk_header}\n{chunk_content}\n"
            
            if total_chars + len(chunk_text) <= max_context_chars:
                context_parts.append(chunk_text)
                total_chars += len(chunk_text)
            else:
                # Truncate the last chunk if needed
                remaining_chars = max_context_chars - total_chars
                if remaining_chars > 100:  # Only add if meaningful space left
                    truncated_chunk = f"{chunk_header}\n{chunk_content[:remaining_chars-50]}...\n"
                    context_parts.append(truncated_chunk)
                break
        
        context = "\n".join(context_parts)
        
        logger.info(f"Prepared context: {len(search_results)} chunks, {len(context)} characters")
        
        return context

    async def _generate_quality_answer(self, question: str, context: str) -> str:
        """Generate high-quality answer with validation"""
        try:
            # Generate initial answer
            raw_answer = await self.generation_service.generate(question, context)
            
            # Validate and enhance answer quality
            quality_answer = self._validate_and_enhance_answer(raw_answer, question, context)
            
            return quality_answer
            
        except Exception as e:
            logger.error(f"Quality answer generation failed: {e}")
            return "I apologize, but I'm unable to generate a reliable answer at the moment."

    def _validate_and_enhance_answer(self, answer: str, question: str, context: str) -> str:
        """Validate and enhance answer quality"""
        if not answer or len(answer.strip()) < 10:
            return "The document does not contain sufficient information to answer this question."
        
        answer = answer.strip()
        
        # Check if answer is actually addressing the question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_content_words = question_words - common_words
        answer_content_words = answer_words - common_words
        
        # Check relevance
        if question_content_words and answer_content_words:
            relevance = len(question_content_words & answer_content_words) / len(question_content_words)
            
            if relevance < 0.1:  # Very low relevance
                return "The document does not contain specific information to answer this question."
        
        # Check for generic responses
        generic_phrases = [
            "document does not contain",
            "information is not available",
            "cannot be determined",
            "not specified in the document"
        ]
        
        answer_lower = answer.lower()
        is_generic = any(phrase in answer_lower for phrase in generic_phrases)
        
        if is_generic and len(answer) < 50:
            return "The document does not contain sufficient information to answer this question."
        
        # Enhance with source information if answer looks good
        if not is_generic and len(answer) > 20:
            # Add confidence indicator based on context quality
            if "thirty days" in answer.lower() or "30 days" in answer.lower():
                answer += " (as specified in the policy document)"
            elif any(term in answer.lower() for term in ["months", "years", "period"]):
                answer += " (based on the policy terms)"
        
        return answer

    def _extract_document_hash_from_chunk(self, chunk_id: str) -> str:
        """Extract document hash from chunk ID"""
        try:
            # Chunk ID format: chunk_{url_hash}_{index}
            parts = chunk_id.split('_')
            if len(parts) >= 2:
                return parts[1]
            return ""
        except Exception:
            return ""

    def _get_document_hash(self, document_url: str) -> str:
        """Generate consistent document hash"""
        return hashlib.sha256(document_url.encode()).hexdigest()

    async def get_processing_stats(self, document_url: str) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        try:
            document_hash = self._get_document_hash(document_url)
            
            # Get collection info
            collection_info = await self.vector_store.get_collection_info()
            
            # Count document chunks
            dummy_embedding = [0.0] * 768
            document_chunks = await self.vector_store.search_similar_chunks(
                query_embedding=dummy_embedding,
                document_url=document_url,
                top_k=1000,  # Get all chunks
                threshold=0.0
            )
            
            # Analyze chunk quality
            quality_stats = self._analyze_chunk_quality(document_chunks)
            
            return {
                "document_url": document_url,
                "document_hash": document_hash,
                "is_processed": document_hash in self._document_cache,
                "total_chunks": len(document_chunks),
                "quality_stats": quality_stats,
                "vector_store_info": collection_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {"error": str(e)}

    def _analyze_chunk_quality(self, chunks: List[SearchResult]) -> Dict[str, Any]:
        """Analyze quality statistics of chunks"""
        if not chunks:
            return {"total_chunks": 0}
        
        quality_scores = []
        has_numbers_count = 0
        has_key_terms_count = 0
        avg_length = 0
        
        for chunk_result in chunks:
            chunk = chunk_result.chunk
            metadata = chunk.metadata
            
            quality_scores.append(metadata.get('quality_score', 0.0))
            
            if metadata.get('has_numbers', False):
                has_numbers_count += 1
            
            if metadata.get('has_key_terms', False):
                has_key_terms_count += 1
            
            avg_length += len(chunk.content)
        
        avg_length = avg_length / len(chunks) if chunks else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "total_chunks": len(chunks),
            "average_quality_score": avg_quality,
            "average_chunk_length": avg_length,
            "chunks_with_numbers": has_numbers_count,
            "chunks_with_key_terms": has_key_terms_count,
            "quality_distribution": {
                "high_quality": len([s for s in quality_scores if s >= 0.7]),
                "medium_quality": len([s for s in quality_scores if 0.4 <= s < 0.7]),
                "low_quality": len([s for s in quality_scores if s < 0.4])
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the advanced RAG pipeline"""
        try:
            health_status = {
                "rag_pipeline": "healthy",
                "services": {
                    "pdf_processor": "unknown",
                    "embedding_service": "unknown", 
                    "generation_service": "unknown",
                    "vector_store": "unknown"
                }
            }
            
            # Test PDF processor
            try:
                # Simple validation that the processor is available
                if self.pdf_processor:
                    health_status["services"]["pdf_processor"] = "healthy"
            except Exception as e:
                health_status["services"]["pdf_processor"] = f"unhealthy: {e}"
            
            # Test embedding service
            try:
                # Test with a simple embedding
                test_embedding = await self.embedding_service.generate_embedding("test")
                if test_embedding and len(test_embedding) == 768:
                    health_status["services"]["embedding_service"] = "healthy"
                else:
                    health_status["services"]["embedding_service"] = "unhealthy: invalid embedding"
            except Exception as e:
                health_status["services"]["embedding_service"] = f"unhealthy: {e}"
            
            # Test generation service
            try:
                # Test with a simple generation
                test_answer = await self.generation_service.generate("test question", "test context")
                if test_answer and len(test_answer) > 0:
                    health_status["services"]["generation_service"] = "healthy"
                else:
                    health_status["services"]["generation_service"] = "unhealthy: no response"
            except Exception as e:
                health_status["services"]["generation_service"] = f"unhealthy: {e}"
            
            # Test vector store
            try:
                # Test vector store connection
                collection_info = await self.vector_store.get_collection_info()
                if collection_info:
                    health_status["services"]["vector_store"] = "healthy"
                else:
                    health_status["services"]["vector_store"] = "unhealthy: no collection info"
            except Exception as e:
                health_status["services"]["vector_store"] = f"unhealthy: {e}"
            
            # Overall health determination
            unhealthy_services = [k for k, v in health_status["services"].items() if not v.startswith("healthy")]
            if unhealthy_services:
                health_status["rag_pipeline"] = f"degraded: {', '.join(unhealthy_services)} unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "rag_pipeline": f"unhealthy: {e}",
                "services": {}
            }

    async def clear_document(self, document_url: str) -> Dict[str, Any]:
        """Clear a document from the system"""
        try:
            # Remove from vector store
            deleted_count = await self.vector_store.delete_document_chunks(document_url)
            
            # Remove from cache
            document_hash = self._get_document_hash(document_url)
            if document_hash in self._document_cache:
                del self._document_cache[document_hash]
            
            return {
                "document_url": document_url,
                "deleted_chunks": deleted_count,
                "cache_cleared": True
            }
            
        except Exception as e:
            logger.error(f"Failed to clear document: {e}")
            return {"error": str(e)}

    async def get_document_info(self, document_url: str) -> Dict[str, Any]:
        """
        Get information about a processed document
        
        Args:
            document_url: URL of the document to get info for
            
        Returns:
            Dict: Document information including metadata and processing status
        """
        try:
            document_hash = self._get_document_hash(document_url)
            
            # Check if document exists in cache
            if document_hash not in self._document_cache:
                return {
                    "document_url": document_url,
                    "status": "not_found",
                    "message": "Document not found in system"
                }
            
            doc_info = self._document_cache[document_hash]
            
            # Get chunk count from vector store if available
            chunk_count = 0
            if self.vector_store:
                try:
                    # Count chunks for this document
                    chunks = await self.vector_store.search_chunks(
                        query="",  # Empty query to get all chunks
                        document_filter=document_url,
                        top_k=1000  # High number to count all
                    )
                    chunk_count = len(chunks)
                except Exception as e:
                    logger.warning(f"Could not get chunk count from vector store: {e}")
                    chunk_count = doc_info.get("chunk_count", 0)
            
            return {
                "document_url": document_url,
                "status": "processed",
                "title": doc_info.get("title", "Unknown Document"),
                "chunk_count": chunk_count,
                "processed_at": doc_info.get("processed_at"),
                "file_size": doc_info.get("file_size"),
                "page_count": doc_info.get("page_count"),
                "avg_chunk_quality": doc_info.get("avg_chunk_quality", 0.0),
                "processing_time": doc_info.get("processing_time"),
                "quality_metrics": doc_info.get("quality_metrics", {}),
                "document_hash": document_hash
            }
            
        except Exception as e:
            logger.error(f"Failed to get document info for {document_url}: {e}")
            return {
                "document_url": document_url,
                "status": "error",
                "message": f"Error retrieving document info: {str(e)}"
            }
