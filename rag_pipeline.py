import asyncio
import hashlib
from typing import List, Dict, Any
from loguru import logger

from pdf_processor import PDFProcessor, TextChunker
from ai_services import EmbeddingService, GenerativeModel
from vector_store import VectorStore
from models import DocumentChunk, SearchResult
from config import get_settings

settings = get_settings()

class RAGPipeline:
    """Main RAG pipeline orchestrating the entire document processing and QA workflow"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.generative_model = GenerativeModel()
        self.vector_store = VectorStore()
        
        # Cache for processed documents (in production, use Redis or database)
        self._document_cache: Dict[str, bool] = {}

    async def process_document_and_answer(self, document_url: str, questions: List[str]) -> List[str]:
        """Main pipeline: process document and answer questions"""
        try:
            logger.info(f"Starting RAG pipeline for document: {document_url}")
            
            # Step 1: Process document if not already processed
            await self._ensure_document_processed(document_url)
            
            # Step 2: Answer questions
            answers = await self._answer_questions(document_url, questions)
            
            logger.info(f"RAG pipeline completed successfully for {len(questions)} questions")
            return answers
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            raise

    async def _ensure_document_processed(self, document_url: str):
        """Ensure document is processed and stored in vector database"""
        document_hash = self._get_document_hash(document_url)
        
        if document_hash in self._document_cache:
            logger.info(f"Document already processed: {document_url}")
            return
        
        logger.info(f"Processing new document: {document_url}")
        
        try:
            # Step 1: Download and extract text
            pdf_content = await self.pdf_processor.download_pdf(document_url)
            text = await self.pdf_processor.extract_text(pdf_content)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Step 2: Create chunks
            chunks = self.text_chunker.create_chunks(text, document_url)
            
            if not chunks:
                raise ValueError("No chunks could be created from the document")
            
            # Step 3: Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.generate_batch_embeddings(chunk_texts)
            
            # Step 4: Store in vector database
            await self.vector_store.store_chunks(chunks, embeddings, document_url)
            
            # Cache the document processing status
            self._document_cache[document_hash] = True
            
            logger.info(f"Document processed successfully: {len(chunks)} chunks stored")
            
        except Exception as e:
            logger.error(f"Failed to process document {document_url}: {e}")
            raise

    async def _answer_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """Answer questions using enhanced RAG approach"""
        answers = []
        document_hash = self._get_document_hash(document_url)
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:100]}...")
                
                # Enhanced retrieval with multiple strategies
                search_results = await self._enhanced_retrieval(question, document_hash)
                
                if not search_results:
                    logger.warning(f"No relevant chunks found for question: {question}")
                    answers.append("The document does not contain information to answer this question.")
                    continue
                
                # Step 3: Prepare enhanced context from retrieved chunks
                context = self._prepare_enhanced_context(search_results, question)
                
                # Step 4: Generate answer using enhanced context
                answer = await self.generative_model.generate_enhanced_answer(question, context)
                answers.append(answer)
                
                logger.info(f"Question {i+1} answered successfully")
                
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {e}")
                answers.append("I apologize, but I encountered an error while processing this question.")
        
        return answers

    async def _enhanced_retrieval(self, question: str, document_hash: str) -> List[SearchResult]:
        """Enhanced retrieval using multiple strategies"""
        try:
            # Strategy 1: Direct semantic search
            question_embedding = await self.embedding_service.generate_embedding(question)
            primary_results = await self.vector_store.search_similar_chunks(
                query_embedding=question_embedding,
                document_url=None,  # Search across all documents first
                top_k=settings.top_k_chunks
            )
            
            # Filter by document hash if we have it
            filtered_results = []
            for result in primary_results:
                result_hash = self._get_document_hash(result.metadata.get('document_url', ''))
                if result_hash == document_hash:
                    filtered_results.append(result)
            
            # Strategy 2: Keyword-based fallback if semantic search yields few results
            if len(filtered_results) < settings.top_k_chunks // 2:
                keyword_results = await self._keyword_search(question, document_hash)
                # Merge and deduplicate
                all_results = filtered_results + keyword_results
                filtered_results = self._deduplicate_results(all_results)
            
            # Strategy 3: Relaxed threshold search if still not enough results
            if len(filtered_results) < 3:
                relaxed_results = await self.vector_store.search_similar_chunks(
                    query_embedding=question_embedding,
                    document_url=None,
                    top_k=settings.top_k_chunks * 2,
                    threshold=settings.similarity_threshold * 0.7  # More relaxed
                )
                # Filter and merge
                for result in relaxed_results:
                    result_hash = self._get_document_hash(result.metadata.get('document_url', ''))
                    if result_hash == document_hash and result not in filtered_results:
                        filtered_results.append(result)
            
            # Sort by relevance and return top results
            filtered_results.sort(key=lambda x: x.score, reverse=True)
            return filtered_results[:settings.top_k_chunks]
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            return []

    async def _keyword_search(self, question: str, document_hash: str) -> List[SearchResult]:
        """Keyword-based search as fallback"""
        # Extract keywords from question
        keywords = self._extract_keywords(question)
        
        # Create a keyword-based query
        keyword_query = " ".join(keywords)
        if keyword_query.strip():
            keyword_embedding = await self.embedding_service.generate_embedding(keyword_query)
            return await self.vector_store.search_similar_chunks(
                query_embedding=keyword_embedding,
                document_url=None,
                top_k=settings.top_k_chunks // 2,
                threshold=settings.similarity_threshold * 0.8
            )
        return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from question"""
        import re
        
        # Remove common question words and short words
        stop_words = {
            'what', 'is', 'are', 'how', 'when', 'where', 'why', 'does', 'do', 'can', 
            'will', 'would', 'should', 'could', 'the', 'a', 'an', 'and', 'or', 'but', 
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'under', 'this', 'that'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Top 5 keywords

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            # Use first 100 characters as uniqueness key
            content_key = result.content[:100].strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results

    def _prepare_enhanced_context(self, search_results: List[SearchResult], question: str) -> str:
        """Prepare enhanced context with better formatting and relevance scoring"""
        if not search_results:
            return ""
        
        # Sort by relevance score
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        # Build context with numbered sections and relevance scores
        context_parts = []
        total_chars = 0
        max_context_chars = settings.context_window
        
        for i, result in enumerate(sorted_results, 1):
            chunk_content = result.content.strip()
            
            # Skip if adding this chunk would exceed context window
            if total_chars + len(chunk_content) > max_context_chars:
                break
            
            # Format with relevance information
            formatted_chunk = f"[Source {i} - Relevance: {result.score:.3f}]\n{chunk_content}"
            context_parts.append(formatted_chunk)
            total_chars += len(formatted_chunk) + 20  # Account for formatting
        
        # Join context parts with clear separators
        context = "\n\n" + "="*50 + "\n\n".join([""] + context_parts) + "\n" + "="*50 + "\n"
        
        logger.info(f"Enhanced context prepared: {len(context_parts)} chunks, {total_chars} characters")
        return context

    def _get_document_hash(self, document_url: str) -> str:
        """Generate hash for document URL for caching"""
        return hashlib.sha256(document_url.encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all pipeline components"""
        health_status = {
            "rag_pipeline": "healthy",
            "services": {}
        }
        
        try:
            # Check vector store
            vector_store_healthy = await self.vector_store.health_check()
            health_status["services"]["vector_store"] = "healthy" if vector_store_healthy else "unhealthy"
            
            # Check if models are loaded
            embedding_healthy = self.embedding_service.model is not None
            health_status["services"]["embedding_service"] = "healthy" if embedding_healthy else "unhealthy"
            
            generative_healthy = (
                self.generative_model.use_openrouter and 
                self.generative_model.api_key is not None
            ) or not self.generative_model.use_openrouter
            health_status["services"]["generative_model"] = "healthy" if generative_healthy else "unhealthy"
            
            # Overall health
            all_healthy = all(
                status == "healthy" 
                for status in health_status["services"].values()
            )
            
            if not all_healthy:
                health_status["rag_pipeline"] = "degraded"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["rag_pipeline"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status

    async def get_document_info(self, document_url: str) -> Dict[str, Any]:
        """Get information about a processed document"""
        try:
            # Get collection info from vector store
            collection_info = await self.vector_store.get_collection_info()
            
            # Check if document is processed
            document_hash = self._get_document_hash(document_url)
            is_processed = document_hash in self._document_cache
            
            return {
                "document_url": document_url,
                "is_processed": is_processed,
                "document_hash": document_hash,
                "vector_store_info": collection_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get document info: {e}")
            return {"error": str(e)}

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
