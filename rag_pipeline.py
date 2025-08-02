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
        """Answer questions using RAG approach"""
        answers = []
        
        for i, question in enumerate(questions):
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}: {question[:100]}...")
                
                # Step 1: Generate question embedding
                question_embedding = await self.embedding_service.generate_embedding(question)
                
                # Step 2: Retrieve relevant chunks
                search_results = await self.vector_store.search_similar_chunks(
                    query_embedding=question_embedding,
                    document_url=document_url
                )
                
                if not search_results:
                    logger.warning(f"No relevant chunks found for question: {question}")
                    answers.append("I could not find relevant information in the document to answer this question.")
                    continue
                
                # Step 3: Prepare context from retrieved chunks
                context = self._prepare_context(search_results, question)
                
                # Step 4: Generate answer using context
                answer = await self.generative_model.generate_answer(question, context)
                answers.append(answer)
                
                logger.info(f"Question {i+1} answered successfully")
                
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {e}")
                answers.append("I apologize, but I encountered an error while processing this question.")
        
        return answers

    def _prepare_context(self, search_results: List[SearchResult], question: str) -> str:
        """Prepare context from search results for answer generation"""
        if not search_results:
            return ""
        
        # Sort by relevance score
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        # Combine context from top results
        context_parts = []
        total_chars = 0
        max_context_chars = settings.context_window
        
        for result in sorted_results:
            chunk_content = result.content.strip()
            
            # Skip if adding this chunk would exceed context window
            if total_chars + len(chunk_content) > max_context_chars:
                # Try to add partial content if possible
                remaining_chars = max_context_chars - total_chars
                if remaining_chars > 100:  # Only add if meaningful portion can be included
                    partial_content = chunk_content[:remaining_chars].rsplit(' ', 1)[0]
                    context_parts.append(partial_content)
                break
            
            context_parts.append(chunk_content)
            total_chars += len(chunk_content)
        
        # Join context parts with clear separators
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Prepared context with {len(context_parts)} chunks, {total_chars} characters")
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
