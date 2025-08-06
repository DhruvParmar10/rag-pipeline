import asyncio
import hashlib
import re
from typing import List, Dict, Any
from loguru import logger

from pdf_processor import PDFProcessor, TextChunker
from ai_services import EmbeddingService, GenerationService
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
        self.generation_service = GenerationService()
        self.vector_store = VectorStore()
        
        # Cache for processed documents (in production, use Redis or database)
        self._document_cache: Dict[str, bool] = {}
        
        # Cache for embeddings to avoid re-computation (aggressive caching)
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Cache for document content to avoid re-downloading
        self._content_cache: Dict[str, str] = {}

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
            logger.info(f"Document already processed (cached): {document_url}")
            return
        
        # Check if document already exists in vector store
        try:
            # Quick search to see if document chunks exist
            dummy_embedding = [0.0] * 768  # Dummy embedding for existence check
            existing_chunks = await self.vector_store.search_similar_chunks(
                query_embedding=dummy_embedding,
                document_url=document_url,
                top_k=1,
                threshold=0.0
            )
            
            if existing_chunks:
                logger.info(f"Document already exists in vector store: {document_url}")
                self._document_cache[document_hash] = True
                return
                
        except Exception as e:
            logger.warning(f"Could not check document existence: {e}")
        
        logger.info(f"Processing new document: {document_url}")
        
        try:
            # Step 1: Download and extract text with caching
            document_hash = self._get_document_hash(document_url)
            
            if document_hash in self._content_cache:
                logger.info("Using cached document content")
                text = self._content_cache[document_hash]
            else:
                pdf_content = await self.pdf_processor.download_pdf(document_url)
                text = await self.pdf_processor.extract_text(pdf_content)
                # Cache the extracted text
                self._content_cache[document_hash] = text
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Step 2: Create chunks with intelligent filtering
            chunks = self.text_chunker.create_chunks(text, document_url)
            
            if not chunks:
                raise ValueError("No chunks could be created from the document")
            
            # Step 2.5: Filter chunks to only embed high-value content
            high_value_chunks = self._filter_high_value_chunks(chunks)
            logger.info(f"Filtered {len(chunks)} chunks to {len(high_value_chunks)} high-value chunks")
            
            # Step 3: Generate embeddings for filtered chunks only
            chunk_texts = [chunk.content for chunk in high_value_chunks]
            embeddings = await self.embedding_service.generate_batch_embeddings(chunk_texts)
            
            # Step 4: Store in vector database
            await self.vector_store.store_chunks(high_value_chunks, embeddings, document_url)
            
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
                answer = await self.generation_service.generate(question, context)
                answers.append(answer)
                
                logger.info(f"Question {i+1} answered successfully")
                
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {e}")
                answers.append("I apologize, but I encountered an error while processing this question.")
        
        return answers

    async def _enhanced_retrieval(self, question: str, document_hash: str) -> List[SearchResult]:
        """Enhanced retrieval optimized for speed and accuracy"""
        try:
            # Check if we have cached embedding for this question
            question_key = hashlib.md5(question.encode()).hexdigest()
            
            if question_key in self._embedding_cache:
                logger.info("Using cached embedding for question")
                question_embedding = self._embedding_cache[question_key]
            else:
                # Strategy 1: Direct semantic search with optimized settings
                question_embedding = await self.embedding_service.generate_embedding(question)
                # Cache the embedding
                self._embedding_cache[question_key] = question_embedding
            
            # Use a more relaxed search first to get more candidates
            primary_results = await self.vector_store.search_similar_chunks(
                query_embedding=question_embedding,
                document_url=None,  # Search across all documents first
                top_k=settings.top_k_chunks * 3,  # Get even more candidates
                threshold=0.0  # Very permissive - get all results
            )
            
            # Filter by document hash if we have it
            filtered_results = []
            logger.info(f"Looking for document hash: {document_hash}")
            for result in primary_results:
                result_url = result.metadata.get('document_url', '')
                result_hash = self._get_document_hash(result_url)
                logger.info(f"Found chunk with URL hash: {result_hash}, score: {result.score:.4f}")
                if result_hash == document_hash:
                    filtered_results.append(result)
                    logger.info(f"✅ Added matching chunk: {result.content[:100]}...")
            
            # If we still don't have enough results, try even more relaxed search
            if len(filtered_results) < 5:
                logger.info(f"Only {len(filtered_results)} results found, trying relaxed search...")
                relaxed_results = await self.vector_store.search_similar_chunks(
                    query_embedding=question_embedding,
                    document_url=None,
                    top_k=settings.top_k_chunks * 5,
                    threshold=0.0  # No threshold - get all results
                )
                # Filter and merge
                for result in relaxed_results:
                    result_url = result.metadata.get('document_url', '')
                    result_hash = self._get_document_hash(result_url)
                    if result_hash == document_hash and result not in filtered_results:
                        filtered_results.append(result)
            
            # If still no results, use any available results for debugging
            if len(filtered_results) == 0 and len(primary_results) > 0:
                logger.warning(f"No document-specific results found. Using top {min(3, len(primary_results))} general results for debugging")
                filtered_results = primary_results[:3]
            
            # Sort by relevance and return top results
            filtered_results.sort(key=lambda x: x.score, reverse=True)
            final_results = filtered_results[:settings.top_k_chunks]
            
            logger.info(f"Retrieved {len(final_results)} relevant chunks for question")
            return final_results
            
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
            embedding_healthy = self.embedding_service.api_key is not None
            health_status["services"]["embedding_service"] = "healthy" if embedding_healthy else "unhealthy"
            
            generative_healthy = (
                self.generation_service.api_key is not None
            )
            health_status["services"]["generation_service"] = "healthy" if generative_healthy else "unhealthy"
            
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

    def _filter_high_value_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """AGGRESSIVELY filter chunks to only include highest-value content for sub-30s embedding"""
        high_value_chunks = []
        
        for chunk in chunks:
            content = chunk.content.lower()
            
            # PRIORITY 1: Always include chunks with critical insurance keywords
            critical_keywords = [
                'coverage', 'benefit', 'policy', 'claim', 'premium', 'deductible',
                'exclusion', 'limitation', 'condition', 'requirement', 'eligible',
                'payment', 'reimbursement', 'waiting period', 'grace period',
                'copay', 'coinsurance', 'out-of-pocket', 'maximum', 'network'
            ]
            
            # PRIORITY 2: Skip low-value content aggressively
            skip_patterns = [
                r'^page \d+', r'^section \d+', r'^chapter \d+', r'^table of contents',
                r'^index$', r'^\d+\s*$', r'^see page \d+', r'^continued on next page',
                r'^for more information', r'^please refer to', r'^as mentioned',
                r'^table \d+', r'^figure \d+', r'^appendix'
            ]
            
            # Check if chunk should be skipped
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, content):
                    should_skip = True
                    break
            
            if should_skip:
                continue
            
            # AGGRESSIVE FILTERING: Only keep chunks with critical content
            has_critical_keywords = any(keyword in content for keyword in critical_keywords)
            is_very_substantial = len(chunk.content) >= 400 and len(chunk.content.split()) >= 50
            
            # Be much more selective - only keep truly important chunks
            if has_critical_keywords or is_very_substantial:
                high_value_chunks.append(chunk)
        
        # NUCLEAR OPTION: Keep maximum 15% of chunks to achieve sub-30s processing
        if len(high_value_chunks) > len(chunks) * 0.15:
            # Sort by importance (critical keywords first, then by length)
            def chunk_score(chunk):
                content = chunk.content.lower()
                keyword_count = sum(1 for keyword in critical_keywords if keyword in content)
                return keyword_count * 1000 + len(chunk.content)  # Prioritize keyword-rich content
            
            high_value_chunks.sort(key=chunk_score, reverse=True)
            target_count = int(len(chunks) * 0.15)  # Keep only 15% max
            high_value_chunks = high_value_chunks[:target_count]
        
        # Ensure we keep at least 10% to not lose critical info
        if len(high_value_chunks) < len(chunks) * 0.10:
            sorted_chunks = sorted(chunks, key=lambda c: len(c.content), reverse=True)
            target_count = max(len(high_value_chunks), int(len(chunks) * 0.10))
            high_value_chunks = sorted_chunks[:target_count]
        
        return high_value_chunks

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
