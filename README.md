---
title: Rag Blaze
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: gradio_app.py
pinned: false
license: mit
---

# RAG Blaze 🔥

A high-performance Retrieval-Augmented Generation (RAG) system for document processing and question answering with both Gradio web interface and FastAPI backend.

## Features

- 📄 **PDF Document Processing** - Fast text extraction and chunking
- 🔥 **Blaze Strategy** - Quick answers for straightforward questions
- 🔍 **Deep Dive Analysis** - Advanced processing for complex queries
- 🎯 **Smart Triage** - Intelligent quality assessment and routing
- 📊 **Ensemble Retrieval** - BM25 + Vector similarity search
- 🤖 **AI-Powered** - OpenRouter/OpenAI integration for responses
- 🎨 **Dual Interface** - Gradio web UI + FastAPI REST API

## How to Use

### Web Interface (Gradio)

1. **Upload a PDF document** using the file uploader
2. **Enter your questions** (one per line, max 10 questions)
3. **Click "Process Document"** to get AI-powered answers
4. **View results** with formatted responses for each question

### API Interface (FastAPI)

The same backend powers a REST API accessible at `/api/v1/hackrx/run`

```bash
curl -X POST "https://your-space.hf.space/api/v1/hackrx/run" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": [
      "What is the main topic?",
      "Summarize key findings"
    ]
  }'
```

## Example Questions

- **Content Analysis**: "What is the main topic of this document?"
- **Summarization**: "Summarize the key findings and recommendations"
- **Specific Information**: "What are the important dates mentioned?"
- **Requirements**: "List the main requirements or criteria"
- **Action Items**: "What are the next steps outlined?"

## Technical Architecture

### Processing Strategy

- **Blaze Mode**: Fast TF-IDF similarity + direct LLM processing
- **Deep Dive Mode**: Ensemble retrieval + document re-ranking + enhanced context
- **Smart Triage**: Automatic quality assessment determines processing path

### Components

- **Document Processing**: PyMuPDF for PDF parsing and text extraction
- **Text Chunking**: Recursive character splitting with configurable overlap
- **Embeddings**: TF-IDF vectorization (lightweight, no external dependencies)
- **Vector Store**: FAISS with SimpleVectorStore fallback
- **Retrieval**: BM25 + Vector similarity ensemble approach
- **LLM Integration**: OpenRouter API with qwen/qwen-2.5-72b-instruct

### Performance Optimizations

- Document size limits (50MB download, 500KB text)
- Chunk limits (200 chunks max)
- Timeout protections (PDF parsing, embeddings)
- Concurrent processing with configurable limits
- Memory-efficient sparse matrix operations

## Environment Variables

Set these in your Hugging Face Space settings:

| Variable             | Description              | Default                    |
| -------------------- | ------------------------ | -------------------------- |
| `BEARER_TOKEN`       | API authentication token | Auto-generated             |
| `OPENROUTER_API_KEY` | OpenRouter API key       | Required                   |
| `GENERATIVE_MODEL`   | Model name               | qwen/qwen-2.5-72b-instruct |
| `CHUNK_SIZE`         | Text chunk size          | 600                        |

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio interface
python gradio_app.py

# Run FastAPI server
python app.py
```

### API Documentation

When running FastAPI mode, visit `/docs` for interactive API documentation.

## Deployment Modes

1. **Hugging Face Spaces**: Gradio interface (this deployment)
2. **FastAPI Server**: REST API for integration
3. **Hybrid Mode**: Both interfaces available

---

**🔥 RAG Blaze combines the best of fast processing and deep analysis for intelligent document Q&A!**
