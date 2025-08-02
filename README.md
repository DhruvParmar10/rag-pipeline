# Advanced RAG-based PDF QA Pipeline

A production-ready, sophisticated PDF Question-Answering system using Retrieval-Augmented Generation (RAG) architecture with Qwen2.5-72B and Qdrant Cloud vector database.

## 🌐 Live Demo

🚀 **Deployed on Vercel**: [https://your-app-name.vercel.app](https://your-app-name.vercel.app)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/rag-pdf-qa-pipeline)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Document  │ -> │  Text Extraction │ -> │   Text Chunking │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Question │ -> │   Embedding Gen  │ -> │  Vector Search  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Final Answer  │ <- │  Answer Generation│ <- │Context Retrieval│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Key Features

### Advanced PDF Processing

- **Multi-method text extraction**: pdfplumber, PyMuPDF, OCR fallback
- **Intelligent chunking**: 500-token chunks with 25% semantic overlap
- **Scanned PDF support**: OCR with Tesseract for image-based documents
- **Robust error handling**: Graceful fallbacks for processing failures

### Sophisticated RAG Pipeline

- **Semantic embedding**: sentence-transformers/all-MiniLM-L6-v2
- **Vector storage**: Qdrant with cosine similarity search
- **Context-aware generation**: Qwen2-1.5B-Instruct model
- **Intelligent retrieval**: Top-K chunks with similarity thresholding

### Production-Ready Features

- **Async architecture**: Full async/await support for scalability
- **Caching layer**: Redis for embedding and result caching
- **Health monitoring**: Comprehensive health checks and metrics
- **Security**: Bearer token authentication
- **Error handling**: Graceful error recovery and meaningful responses

## 🛠️ Technical Stack

| Component            | Technology            | Purpose                     |
| -------------------- | --------------------- | --------------------------- |
| **Web Framework**    | FastAPI               | High-performance async API  |
| **Vector Database**  | Qdrant                | Semantic search and storage |
| **Embedding Model**  | sentence-transformers | Text to vector conversion   |
| **Generative Model** | Qwen2-1.5B-Instruct   | Answer generation           |
| **PDF Processing**   | pdfplumber, PyMuPDF   | Text extraction             |
| **OCR Engine**       | Tesseract             | Scanned document processing |
| **Caching**          | Redis                 | Performance optimization    |
| **Deployment**       | Docker                | Containerized deployment    |

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Qdrant and Redis)
- 8GB+ RAM (for models)
- CUDA-compatible GPU (optional, for acceleration)

### 1. Setup

```bash
# Clone and setup
git clone <repository>
cd LmaoTeam

# Run automated setup
./setup.sh
```

### 2. Start Services

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 qdrant/qdrant

# Start Redis (optional, for caching)
docker run -d -p 6379:6379 redis:alpine

# Start the RAG pipeline server
./start_server.sh
```

### 3. Test the System

```bash
# Run comprehensive tests
./run_tests.sh
```

## 📡 API Usage

### Main Endpoint

```http
POST /api/v1/hackrx/run
Content-Type: application/json
Authorization: Bearer 7c695e780a6ab6eacffab7c9326e5d8e472a634870a6365979c5671ad28f003c

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What are the coverage details for maternity expenses?"
  ]
}
```

### Response Format

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment...",
    "The policy covers maternity expenses including childbirth..."
  ]
}
```

### Additional Endpoints

- `GET /health` - System health check
- `GET /api/v1/stats` - System statistics
- `GET /api/v1/document/info?document_url=<url>` - Document processing status
- `DELETE /api/v1/document/clear?document_url=<url>` - Clear document from system

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GENERATIVE_MODEL=Qwen/Qwen2-1.5B-Instruct
EMBEDDING_DIMENSION=384

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=pdf_documents

# Processing Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=0.25
TOP_K_CHUNKS=5
SIMILARITY_THRESHOLD=0.7
CONTEXT_WINDOW=2000
```

## 🏗️ System Architecture

### Core Components

1. **PDF Processor** (`pdf_processor.py`)

   - Multi-method text extraction with fallbacks
   - Intelligent text chunking with overlap
   - OCR support for scanned documents

2. **AI Services** (`ai_services.py`)

   - Embedding generation with caching
   - Text generation using Qwen model
   - Batch processing optimization

3. **Vector Store** (`vector_store.py`)

   - Qdrant integration for semantic search
   - Efficient chunk storage and retrieval
   - Collection management

4. **RAG Pipeline** (`rag_pipeline.py`)
   - Orchestrates entire workflow
   - Document processing and caching
   - Question answering with context

### Data Flow

1. **Document Ingestion**

   ```
   PDF URL → Download → Text Extraction → Chunking → Embedding → Vector Storage
   ```

2. **Question Answering**
   ```
   Question → Embedding → Vector Search → Context Retrieval → Answer Generation
   ```

## 🚀 Performance Optimizations

### Caching Strategy

- **Embedding Cache**: Redis-based caching for generated embeddings
- **Document Cache**: In-memory tracking of processed documents
- **Model Loading**: Lazy loading and singleton pattern

### Memory Management

- **Streaming Processing**: Large documents processed in chunks
- **GPU Optimization**: Automatic CUDA detection and utilization
- **Batch Operations**: Efficient batch embedding generation

### Scalability Features

- **Async Operations**: Non-blocking I/O for concurrent requests
- **Connection Pooling**: Efficient database connections
- **Background Tasks**: Async document processing

## 🛡️ Security & Error Handling

### Security Measures

- Bearer token authentication
- Input validation and sanitization
- Rate limiting (configurable)
- CORS configuration for cross-origin requests

### Error Handling

- Graceful PDF processing failures
- Model loading error recovery
- Network timeout handling
- Meaningful error responses

### Monitoring & Logging

- Structured logging with loguru
- Health check endpoints
- Performance metrics collection
- Error tracking and reporting

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the application
docker build -t rag-pipeline .

# Run with docker-compose
docker-compose up -d
```

### Docker Compose Setup

```yaml
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - redis

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 📊 Performance Benchmarks

### Model Performance

- **Embedding Generation**: ~50ms per chunk (CPU), ~10ms (GPU)
- **Answer Generation**: ~2-5s per question (depending on context)
- **Document Processing**: ~30s for 50-page PDF
- **Vector Search**: <100ms for 10K chunks

### System Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, NVIDIA GPU
- **Storage**: 10GB for models + data

## 🧪 Testing

### Test Coverage

- Unit tests for each component
- Integration tests for full pipeline
- Performance benchmarks
- Error scenario testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python test_rag_pipeline.py

# Load testing
python load_test.py
```

## 🚀 Deployment

### Deploy to Vercel

1. **Fork this repository** to your GitHub account

2. **Connect to Vercel**:

   - Go to [vercel.com](https://vercel.com)
   - Import your forked repository
   - Vercel will automatically detect the FastAPI app

3. **Configure Environment Variables** in Vercel Dashboard:

   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   AUTH_TOKEN=your_secure_auth_token
   USE_OPENROUTER=true
   GENERATIVE_MODEL=qwen/qwen-2.5-72b-instruct
   QDRANT_COLLECTION_NAME=pdf_documents
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   TOP_K_CHUNKS=8
   SIMILARITY_THRESHOLD=0.6
   API_TIMEOUT=25
   LOG_LEVEL=INFO
   ```

4. **Deploy**: Vercel will automatically build and deploy your app

### Manual Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
vercel

# Set environment variables
vercel env add OPENROUTER_API_KEY
vercel env add QDRANT_URL
vercel env add QDRANT_API_KEY
# ... add other variables

# Deploy to production
vercel --prod
```

### Local Development with Production Settings

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Start development server
uvicorn main:app --reload --port 8000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent language models
- **Qdrant** for the high-performance vector database
- **Hugging Face** for the transformers ecosystem
- **FastAPI** for the modern web framework

---

**Built with ❤️ for production-ready AI applications**
