import os
import requests
import tempfile
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, APIRouter
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
import hashlib
import json
from datetime import datetime

# Import LangChain components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings

# Import for TF-IDF embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Simple TF-IDF Embeddings class - optimized for speed
class TfidfEmbeddings(Embeddings):
    """Simple TF-IDF based embeddings using scikit-learn - optimized for performance"""
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = None
        self.fitted = False
        self._corpus = []
    
    def _ensure_fitted(self, texts: List[str]):
        """Ensure the vectorizer is fitted on a corpus"""
        if not self.fitted:
            # Combine with any existing corpus
            all_texts = self._corpus + texts
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 1),  # Reduced from (1, 2) for speed
                min_df=1,
                max_df=0.95,  # Reduced from 0.95
                norm='l2',  # Explicit normalization
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True  # Use sublinear tf scaling for better performance
            )
            self.vectorizer.fit(all_texts)
            self.fitted = True
            self._corpus = all_texts
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using TF-IDF"""
        self._ensure_fitted(texts)
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Convert sparse matrix to dense more efficiently
        return tfidf_matrix.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using TF-IDF"""
        if not self.fitted:
            # If not fitted yet, return a zero vector
            return [0.0] * self.max_features
        
        tfidf_vector = self.vectorizer.transform([text])
        return tfidf_vector.toarray().flatten().tolist()

# --- Initial Setup & Global Objects ---
load_dotenv()

print("Loading model with high-accuracy parameters...")
llm_model = ChatOpenAI(
    model=os.getenv("GENERATIVE_MODEL", "qwen/qwen-2.5-72b-instruct"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_tokens=200,  # Reduced from 400
    top_p=0.9,
    default_headers={"HTTP-Referer": "http://localhost"}
)
print("✅ Advanced model loaded successfully.")

embeddings = TfidfEmbeddings(max_features=500)  # Reduced from 1000
print("✅ TF-IDF embeddings initialized (no external dependencies).")

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str] = Field(..., min_items=1, max_items=20)

class HackRxResponse(BaseModel):
    answers: List[str]

class RerankScore(BaseModel):
    score: int = Field(..., description="The relevance score from 0 to 10.")

app = FastAPI(
    title="Advanced RAG API - Blaze & Deep Dive Strategy",
    description="Enterprise-grade RAG system with intelligent triage and deep dive processing",
    version="2.0.0"
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
router = APIRouter(prefix="/api/v1")
security = HTTPBearer()
API_KEY = os.getenv("BEARER_TOKEN", "7c695e780a6ab6eacffab7c9326e5d8e472a634870a6365979c5671ad28f003c")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.post("/hackrx/run", response_model=HackRxResponse)
def run_hackrx_job(
    request: HackRxRequest,
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Advanced RAG endpoint with Blaze & Deep Dive strategy:
    - Fast "Blaze" processing for simple questions
    - Intelligent triage to identify questions needing deep dive
    - Deep dive with ensemble retrieval and re-ranking for complex questions
    """
    start_time = datetime.now()
    
    try:
        doc_url = str(request.documents)
        print(f"🎯 Processing document with Blaze & Deep Dive strategy: {doc_url}")
        
        # Download and process document with timeout and size limits
        print("📥 Downloading document...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            try:
                response = requests.get(doc_url, timeout=15, stream=True)  # Reduced timeout
                response.raise_for_status()
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 50 * 1024 * 1024:  # 50MB limit
                    raise ValueError("Document too large (>50MB)")
                
                # Write content with progress
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        total_size += len(chunk)
                        if total_size > 50 * 1024 * 1024:  # 50MB limit
                            raise ValueError("Document too large (>50MB)")
                
                temp_file_path = temp_file.name
                print(f"✅ Downloaded {total_size // 1024}KB")
                
            except requests.exceptions.Timeout:
                raise HTTPException(status_code=408, detail="Document download timeout")
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        
        # Load and chunk document with timeout protection
        print("📄 Parsing PDF...")
        try:
            import threading
            import time
            
            # Use threading-based timeout instead of signal
            parsing_complete = threading.Event()
            parsing_result = {"docs": None, "error": None}
            
            def parse_pdf():
                try:
                    loader = PyMuPDFLoader(temp_file_path)
                    docs = loader.load()
                    parsing_result["docs"] = docs
                except Exception as e:
                    parsing_result["error"] = e
                finally:
                    parsing_complete.set()
            
            # Start parsing in a separate thread
            parse_thread = threading.Thread(target=parse_pdf)
            parse_thread.daemon = True
            parse_thread.start()
            
            # Wait for completion or timeout (30 seconds)
            if parsing_complete.wait(timeout=30):
                if parsing_result["error"]:
                    raise parsing_result["error"]
                docs = parsing_result["docs"]
            else:
                raise TimeoutError("PDF parsing timeout")
            
            # Check if document is too large (limit to reasonable size)
            total_text_length = sum(len(doc.page_content) for doc in docs)
            if total_text_length > 500000:  # 500KB text limit
                print(f"⚠️ Large document detected ({total_text_length} chars), truncating...")
                # Keep only first portion to prevent timeout
                truncated_docs = []
                current_length = 0
                for doc in docs:
                    if current_length + len(doc.page_content) > 300000:  # 300KB limit
                        break
                    truncated_docs.append(doc)
                    current_length += len(doc.page_content)
                docs = truncated_docs
                print(f"✅ Truncated to {len(docs)} pages ({current_length} chars)")
            else:
                print(f"✅ Parsed {len(docs)} pages ({total_text_length} chars)")
            
        except TimeoutError:
            raise HTTPException(status_code=408, detail="PDF parsing timeout - document too complex")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")
        finally:
            # Always cleanup temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        print("🔪 Chunking document...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 600)),  # Reduced from 800 
            chunk_overlap=int(int(os.getenv("CHUNK_SIZE", 600)) * 0.2)  # Reduced overlap
        )
        
        try:
            chunks = text_splitter.split_documents(docs)
            
            # Limit number of chunks to prevent memory issues and timeouts
            max_chunks = 200  # Reasonable limit for processing speed
            if len(chunks) > max_chunks:
                print(f"⚠️ Too many chunks ({len(chunks)}), limiting to {max_chunks}")
                chunks = chunks[:max_chunks]
            
            print(f"✅ Created {len(chunks)} chunks")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Document chunking failed: {str(e)}")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document processing resulted in zero chunks - document may be empty or corrupted")

        # Create vector store with progress monitoring
        print("🔍 Creating embeddings...")
        try:
            import threading
            
            # Use threading-based timeout for embeddings
            embedding_complete = threading.Event()
            embedding_result = {"vectorstore": None, "error": None}
            
            def create_embeddings():
                try:
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    embedding_result["vectorstore"] = vectorstore
                except Exception as e:
                    embedding_result["error"] = e
                finally:
                    embedding_complete.set()
            
            # Start embedding creation in a separate thread
            embed_thread = threading.Thread(target=create_embeddings)
            embed_thread.daemon = True
            embed_thread.start()
            
            # Wait for completion or timeout (60 seconds)
            if embedding_complete.wait(timeout=60):
                if embedding_result["error"]:
                    raise embedding_result["error"]
                vectorstore = embedding_result["vectorstore"]
            else:
                raise TimeoutError("Embedding creation timeout")
            
            print(f"✅ Vector store created with {len(chunks)} embeddings")
            
        except TimeoutError:
            raise HTTPException(status_code=408, detail="Embedding creation timeout - too many chunks")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector store creation failed: {str(e)}")
        
        # Enhanced RAG prompt for insurance documents
        RAG_PROMPT = PromptTemplate.from_template("""You are an expert insurance policy analyst. Using ONLY the provided context, answer the question with precise, factual information.

INSTRUCTIONS:
- Provide specific details including numbers, timeframes, and conditions
- If multiple conditions apply, list them clearly
- If information is not in the context, state "The document does not contain this information"
- Be concise but comprehensive
- Focus on actionable information

Context:
{context}

Question: {question}

Answer:""")

        # --- Triage Pipeline ---
        triage_prompt = PromptTemplate.from_template("""You are a quality control AI. Given a question and a proposed answer, your task is to determine if the answer is a good, informative answer or if it's a refusal/hallucination.
Respond with only the word 'GOOD' if the answer is specific and informative.
Respond with only the word 'BAD' if the answer is a refusal (e.g., "I cannot answer", "not in the context") or a clear hallucination.

Question: {question}
Answer: {answer}
""")
        triage_chain = triage_prompt | llm_model | StrOutputParser()

        # --- Deep Dive Re-ranker ---
        json_parser = JsonOutputParser(pydantic_object=RerankScore)
        rerank_prompt = PromptTemplate(
            template="Strictly evaluate the relevance of the document to the question on a scale from 0 to 10. Respond with ONLY a JSON object with a single integer key \"score\".\n{format_instructions}\n\nQuestion: {question}\nDocument: {context}", 
            input_variables=["question", "context"], 
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
        )
        rerank_chain = rerank_prompt | llm_model | json_parser

        # --- Create Chains ---
        blaze_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})  # Reduced from 5
        blaze_chain = ({"context": blaze_retriever, "question": RunnablePassthrough()}) | RAG_PROMPT | llm_model | StrOutputParser()

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 15  # Reduced from 25
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})  # Reduced from 25
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.7, 0.3])
        
        # --- EXECUTION LOGIC ---
        final_answers = {}
        questions_to_process = request.questions

        print("📊 PASS 1: Running 'Blaze' answers...")
        blaze_answers = blaze_chain.batch(questions_to_process, {"max_concurrency": 15})  # Increased from 10
        
        triage_inputs = [{"question": q, "answer": a} for q, a in zip(questions_to_process, blaze_answers)]
        triage_results = triage_chain.batch(triage_inputs, {"max_concurrency": 15})  # Increased from 10
        
        questions_for_deep_dive = []
        for i, result in enumerate(triage_results):
            question = questions_to_process[i]
            if "GOOD" in result.upper():
                final_answers[question] = blaze_answers[i]
                print(f"✅ Blaze success for Q{i+1}")
            else:
                questions_for_deep_dive.append(question)
                print(f"🔄 Deep dive needed for Q{i+1}")

        if questions_for_deep_dive:
            print(f"🚀 PASS 2: Running 'Deep Dive' on {len(questions_for_deep_dive)} difficult questions...")
            
            # Step 1: Retrieve documents for all difficult questions
            retrieved_docs = ensemble_retriever.batch(questions_for_deep_dive, {"max_concurrency": 8})  # Increased from 5

            # Step 2: Re-rank documents for each question
            final_contexts = []
            for i, docs_list in enumerate(retrieved_docs):
                question = questions_for_deep_dive[i]
                if not docs_list:
                    final_contexts.append("") # Handle case with no retrieved docs
                    continue
                
                rerank_inputs = [{"question": question, "context": doc.page_content} for doc in docs_list]
                scores = rerank_chain.batch(rerank_inputs, {"max_concurrency": 15})  # Increased from 10
                
                scored_docs = []
                for j, score_dict in enumerate(scores):
                    try:
                        score = score_dict.get('score')
                        if score is not None and score >= 6:  # Increased threshold from 5 to 6
                            scored_docs.append((docs_list[j], score))
                    except (AttributeError, ValueError):
                        continue
                
                if not scored_docs:
                    # Safety Net: use top 2 original docs if re-ranking yields nothing
                    top_docs = docs_list[:2]  # Reduced from 3
                else:
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    top_docs = [doc for doc, score in scored_docs[:3]]  # Reduced from 5

                # Combine the page content of the top documents into a single context string
                final_contexts.append("\n\n---\n\n".join([doc.page_content for doc in top_docs]))

            # Step 3: Generate final answers using the curated contexts
            deep_dive_rag_chain = RAG_PROMPT | llm_model | StrOutputParser()
            deep_dive_inputs = [{"context": c, "question": q} for c, q in zip(final_contexts, questions_for_deep_dive)]
            deep_dive_answers = deep_dive_rag_chain.batch(deep_dive_inputs, {"max_concurrency": 8})  # Increased from 5

            for question, answer in zip(questions_for_deep_dive, deep_dive_answers):
                final_answers[question] = answer

        # Re-order the answers to match the original question order
        ordered_answers = [final_answers.get(q, "An error occurred processing this question.") for q in request.questions]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Blaze & Deep Dive Job Completed in {processing_time:.2f}s")
        print(f"📈 Performance: {len(request.questions) - len(questions_for_deep_dive)}/{len(request.questions)} blaze, {len(questions_for_deep_dive)} deep dive")
        
        return HackRxResponse(answers=ordered_answers)

    except Exception as e:
        import traceback
        print(f"❌ An error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "strategy": "blaze_and_deep_dive",
        "version": "2.0.0"
    }

@router.get("/stats")
def get_system_stats():
    """Get system statistics"""
    return {
        "system_info": {
            "version": "2.0.0",
            "strategy": "blaze_and_deep_dive",
            "features": [
                "fast_blaze_processing",
                "intelligent_triage",
                "ensemble_retrieval",
                "document_reranking",
                "deep_dive_processing"
            ]
        },
        "timestamp": datetime.now().isoformat()
    }

app.include_router(router)

# Export the app for Vercel (this is required)
app = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Advanced RAG API - Blaze & Deep Dive Strategy on port {port}")
    print("🎯 Features: Fast Blaze processing, Intelligent triage, Deep dive with ensemble retrieval")
    print("📊 Focus: Speed for simple questions, accuracy for complex ones")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=port,
        reload=False
    )
