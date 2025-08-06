# RAG Pipeline API

Production-ready Retrieval-Augmented Generation (RAG) system for PDF document processing and intelligent question answering.

## Features

- **Cloud-based Architecture**: Hosted embedding models and vector storage
- **PDF Processing**: Text extraction with OCR support
- **Smart Chunking**: Sentence-aware text segmentation
- **Vector Search**: Qdrant cloud integration for similarity matching
- **Real-time QA**: Fast document query processing

## Project Structure

```
rag-pipeline/
├── main.py                 # FastAPI application entry point
├── rag_pipeline.py          # Core RAG orchestration logic
├── ai_services.py           # Embedding and generation services
├── vector_store.py          # Qdrant vector database interface
├── pdf_processor.py         # PDF text extraction and chunking
├── models.py                # Pydantic data models
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── tests/
    ├── test_qdrant_cloud.py
    ├── test_hf_embedding.py
    └── test_advanced_rag.py
```

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd rag-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Setup

Create `.env` file:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_key
QDRANT_API_KEY=your_qdrant_key

# Model Configuration
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768
GENERATIVE_MODEL=qwen/qwen-2.5-72b-instruct

# Qdrant Configuration
QDRANT_HOST=https://your-cluster.qdrant.io:6333
QDRANT_COLLECTION_NAME=pdf_documents_768

# Security
BEARER_TOKEN=your_secure_token
```

### 3. Run Server

```bash
python main.py
```

Server starts at `http://localhost:8000`

## API Usage

### Process Document & Answer Questions

**Request:**

```http
POST /api/v1/hackrx/run
Authorization: Bearer your_bearer_token
Content-Type: application/json

{
  "document_url": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered?"
  ]
}
```

**Response:**

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
  ]
}
```

### Health Check

```http
GET /health
```

## Embedding Models

### Current: 768-dimension Gemini model (High Quality)

```env
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768
QDRANT_COLLECTION_NAME=pdf_documents_768
```

### Alternative: Free Hugging Face models

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
QDRANT_COLLECTION_NAME=pdf_documents_384
```

⚠️ **Note**: Changing embedding models requires recreating the Qdrant collection with matching dimensions.

## Performance

- **Initial processing**: ~2-3 minutes (first time)
- **Subsequent queries**: ~10-20 seconds
- **Supported formats**: PDF (text and scanned images)
- **Max file size**: 50MB

## Core Components

| Component          | Purpose                             |
| ------------------ | ----------------------------------- |
| `main.py`          | FastAPI application and routing     |
| `rag_pipeline.py`  | Document processing orchestration   |
| `ai_services.py`   | Cloud embedding and generation APIs |
| `vector_store.py`  | Qdrant vector database operations   |
| `pdf_processor.py` | PDF parsing and text chunking       |
| `config.py`        | Environment and settings management |

## Dependencies

- **FastAPI**: REST API framework
- **Qdrant**: Vector database
- **Google Generative AI**: Gemini embedding models
- **OpenRouter**: Text generation
- **PDFPlumber**: PDF text extraction

## Testing

```bash
# Test embedding service
python test_hf_embedding.py

# Test vector database
python test_qdrant_cloud.py

# Full pipeline test
python test_advanced_rag.py
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
```

### Production Environment

```env
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
```

## License

MIT License
