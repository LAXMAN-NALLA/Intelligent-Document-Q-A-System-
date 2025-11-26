# VBRAG - Vector-Based Retrieval Augmented Generation System

A sophisticated RAG (Retrieval Augmented Generation) application that enables question-answering over PDF documents using Google's Gemini AI model. This project demonstrates how to convert documents into vector embeddings, store them efficiently, and retrieve relevant context to generate accurate, source-attributed answers.

## 📋 Table of Contents

- [Overview](#overview)
- [What is RAG?](#what-is-rag)
- [Architecture](#architecture)
- [Technology Stack & Why](#technology-stack--why)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

VBRAG is a document Q&A system that combines:
- **Document Processing**: Extracts and chunks PDF documents
- **Vector Embeddings**: Converts text into numerical representations
- **Semantic Search**: Finds relevant document sections using similarity search
- **AI-Powered Answers**: Generates contextual answers using Google Gemini

**Key Features:**
- ✅ Interactive Web UI interface (Streamlit)
- ✅ Persistent vector storage (ChromaDB)
- ✅ Metadata tracking (page numbers, file names)
- ✅ MMR (Maximum Marginal Relevance) retrieval for diverse results
- ✅ Source attribution with page references
- ✅ Error handling and validation
- ✅ Caching for faster subsequent loads

---

## 🧠 What is RAG?

**Retrieval Augmented Generation (RAG)** is a technique that enhances LLM responses by:

1. **Retrieval**: Finding relevant information from a knowledge base (your documents)
2. **Augmentation**: Adding that context to the user's question
3. **Generation**: Using the LLM to generate an answer based on both the question and retrieved context

### Why RAG?

- **Overcomes LLM limitations**: LLMs have knowledge cutoffs and can't access private documents
- **Reduces hallucinations**: Answers are grounded in actual document content
- **Source attribution**: You can verify answers by checking source documents
- **Domain-specific**: Works with any document set without retraining

### RAG Workflow

```
Document → Chunking → Embeddings → Vector Store
                                           ↓
User Question → Embedding → Similarity Search → Retrieve Relevant Chunks
                                                         ↓
                                              LLM (Question + Context) → Answer
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────┐              ┌──────────────┐            │
│  │              Web UI (main1.py) │            │
│  └──────────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Document Loader → Text Splitter → Embeddings        │   │
│  │  → Vector Store → Retriever → RAG Chain              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data & AI Services                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  ChromaDB    │  │  HuggingFace │  │  Google      │     │
│  │  (Vector DB) │  │  Embeddings  │  │  Gemini AI   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**
   - PDF file → PyPDFLoader extracts text
   - Text → RecursiveCharacterTextSplitter creates chunks
   - Chunks → Metadata added (page numbers, file names)

2. **Vectorization**
   - Text chunks → HuggingFace Embeddings model
   - Chunks → Vector embeddings (384-dimensional vectors)
   - Embeddings → Stored in ChromaDB with metadata

3. **Query Processing**
   - User question → Converted to embedding
   - Similarity search → Find top-k similar chunks (MMR algorithm)
   - Retrieved chunks + Question → Sent to Gemini
   - Gemini → Generates answer with context

---

## 🛠️ Technology Stack & Why

### Core Libraries

#### 1. **LangChain** (`langchain`, `langchain-community`, `langchain-google-genai`)
**Why:** 
- **Abstraction Layer**: Provides unified interface for different LLMs, vector stores, and document loaders
- **RAG Components**: Pre-built chains (RetrievalQA) that handle the complex RAG pipeline
- **Modularity**: Easy to swap components (e.g., change LLM or vector store)
- **Best Practices**: Implements proven patterns for document processing and retrieval

**What it does:**
- Manages document loaders, text splitters, embeddings, vector stores
- Provides RetrievalQA chain that orchestrates retrieval + generation
- Handles prompt templating and LLM interactions

#### 2. **ChromaDB** (`chromadb`)
**Why:**
- **Lightweight**: Embedded database, no separate server needed
- **Fast**: Optimized for similarity search operations
- **Persistent**: Can save vector stores to disk
- **Metadata Support**: Stores page numbers, file names alongside vectors
- **Open Source**: Free and actively maintained

**What it does:**
- Stores vector embeddings in a searchable format
- Performs similarity search (cosine similarity, Euclidean distance)
- Manages metadata for source attribution

#### 3. **HuggingFace Embeddings** (`langchain-huggingface`, `sentence-transformers`)
**Why:**
- **Free & Local**: No API costs, runs on your machine
- **High Quality**: `all-MiniLM-L6-v2` is a proven, efficient model
- **Fast**: Optimized for production use
- **Multilingual**: Supports multiple languages
- **384 Dimensions**: Good balance between quality and speed

**What it does:**
- Converts text chunks into 384-dimensional vector embeddings
- These vectors capture semantic meaning (similar concepts = similar vectors)
- Enables semantic search (finding conceptually similar text)

#### 4. **Google Gemini** (`langchain-google-genai`, `google-generativeai`)
**Why:**
- **State-of-the-Art**: Gemini 2.0 Flash is one of the best LLMs available
- **Fast**: Optimized for speed and efficiency
- **Cost-Effective**: Competitive pricing
- **Context Handling**: Excellent at understanding and using provided context
- **Multimodal**: Can be extended to handle images, audio (future)

**What it does:**
- Takes user question + retrieved document chunks
- Generates coherent, context-aware answers
- Follows instructions in the prompt template

#### 5. **Streamlit** (`streamlit`)
**Why:**
- **Rapid Development**: Build web UIs with minimal code
- **Interactive**: Real-time updates, file uploads, dynamic content
- **User-Friendly**: Clean, modern interface out of the box
- **Python-Native**: No HTML/CSS/JavaScript needed
- **Session Management**: Built-in state management for caching

**What it does:**
- Provides web interface for uploading PDFs
- Displays processing statistics and answers
- Manages user interactions and file uploads

#### 6. **PyPDF** (`pypdf`)
**Why:**
- **Reliable**: Well-established PDF parsing library
- **Text Extraction**: Accurately extracts text from PDFs
- **Metadata**: Preserves page numbers and document structure
- **Lightweight**: Minimal dependencies

**What it does:**
- Extracts text content from PDF files
- Maintains page boundaries for source attribution

#### 7. **python-dotenv** (`python-dotenv`)
**Why:**
- **Security**: Keeps API keys out of code
- **Convenience**: Easy environment variable management
- **Best Practice**: Standard way to handle secrets

**What it does:**
- Loads environment variables from `.env` file
- Securely manages API keys

---

## 🔧 Implementation Details

### 1. Document Processing Pipeline

```python
PDF File → PyPDFLoader → List[Document] → Text Splitter → List[Chunk]
```

**Chunking Strategy:**
- **Chunk Size**: 1000 characters
  - *Why*: Large enough to preserve context, small enough for efficient processing
  - *Balance*: Too small = lost context, too large = irrelevant information
  
- **Chunk Overlap**: 200 characters
  - *Why*: Prevents information loss at chunk boundaries
  - *Example*: If a sentence spans two chunks, overlap ensures it's captured

- **Separators**: `['\n\n', '\n', '. ', ' ', '']`
  - *Why*: Hierarchical splitting preserves document structure
  - *Order*: Tries paragraphs first, then sentences, then words

### 2. Embedding Generation

```python
Text Chunk → HuggingFace Model → 384-dim Vector
```

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384 (compact but effective)
- **Training**: Trained on 1B+ sentence pairs
- **Speed**: ~5000 sentences/second on CPU
- **Quality**: Strong semantic understanding

**How Embeddings Work:**
- Similar concepts → Similar vectors (high cosine similarity)
- Example: "machine learning" and "artificial intelligence" → vectors close in space
- Enables semantic search (not just keyword matching)

### 3. Vector Storage (ChromaDB)

```python
Embeddings + Metadata → ChromaDB → Persistent Storage
```

**Storage Structure:**
- **Vectors**: 384-dimensional arrays
- **Metadata**: `{page: int, source_file: str}`
- **IDs**: Unique identifiers for each chunk

**Persistence:**
- Vector stores saved to `./chroma_db_<hash>/`
- Same file = instant load (no reprocessing)
- Hash-based naming prevents conflicts

### 4. Retrieval Strategy: MMR

**Maximum Marginal Relevance (MMR)** balances:
- **Relevance**: How similar to the query
- **Diversity**: How different from already selected chunks

**Parameters:**
- `k=6`: Return 6 chunks
- `fetch_k=20`: Consider 20 candidates
- `lambda_mult=0.5`: 50% relevance, 50% diversity

**Why MMR?**
- Prevents redundant information
- Provides multiple perspectives
- Better coverage of the document

**Alternative**: Simple similarity search (just relevance, no diversity)

### 5. RAG Chain (RetrievalQA)

```python
Question → Retriever → Context Chunks → Prompt Template → LLM → Answer
```

**Components:**
1. **Retriever**: Finds relevant chunks using MMR
2. **Prompt Template**: Formats question + context
3. **LLM**: Generates answer (Gemini 2.0 Flash)
4. **Chain**: Orchestrates the entire flow

**Prompt Template:**
```
Context: {retrieved_chunks}
Question: {user_question}
Helpful Answer: {llm_response}
```

**Why Custom Prompt?**
- Instructs LLM to use only provided context
- Prevents hallucinations
- Ensures source-based answers

### 6. Answer Generation

**Process:**
1. User question converted to embedding
2. Similarity search finds top-k chunks
3. Chunks + question sent to Gemini
4. Gemini generates answer using context
5. Answer + source metadata returned

**Temperature Setting**: `0.3`
- *Why*: Lower temperature = more focused, deterministic answers
- *Balance*: Too low = repetitive, too high = creative but less accurate

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/))

### Step-by-Step Setup

1. **Clone or navigate to the project directory**
   ```bash
   cd VBRAG
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```
   
   **Important**: Never commit `.env` to version control!

5. **Verify installation**
   ```bash
   python -c "import streamlit; import langchain_google_genai; print('✓ All imports successful!')"
   ```

---

## 🚀 Usage

### Web Interface (`main1.py`)

**For**: User-friendly interface, multiple documents, visual feedback

```bash
streamlit run main1.py
```

**Features:**
- Upload any PDF file
- Visual processing statistics
- Interactive Q&A interface
- Source references with page numbers
- Persistent vector storage (caching)

**Workflow:**
1. Open browser (usually `http://localhost:8501`)
2. Upload PDF file
3. Wait for processing (shows progress)
4. Ask questions
5. View answers with source citations

---

## ⚙️ Configuration

### Key Parameters (in code)

**Chunking:**
```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
```

**Retrieval:**
```python
NUM_RETRIEVED_CHUNKS = 6  # Number of chunks to retrieve
fetch_k = 20              # Candidates for MMR
lambda_mult = 0.5        # MMR diversity (0.0-1.0)
```

**LLM:**
```python
temperature = 0.3        # Lower = more focused
model = "gemini-2.0-flash"
```

### Tuning Guidelines

**For Longer Documents:**
- Increase `CHUNK_SIZE` to 1500-2000
- Increase `NUM_RETRIEVED_CHUNKS` to 8-10

**For Technical Documents:**
- Decrease `CHUNK_SIZE` to 500-800 (preserve code/formulas)
- Increase `CHUNK_OVERLAP` to 300

**For Better Diversity:**
- Decrease `lambda_mult` to 0.3 (more diverse)
- Increase `fetch_k` to 30

**For More Focused Answers:**
- Increase `lambda_mult` to 0.7 (more relevant)
- Decrease `temperature` to 0.1

---

## 🔄 How It Works: Step-by-Step

### Example: "What is the encoder?"

1. **Document Processing** (one-time)
   ```
   article.pdf (10 pages)
   → Extract text: "The encoder is a neural network component..."
   → Split into 45 chunks
   → Generate embeddings: [0.23, -0.45, 0.67, ...] (384 dims)
   → Store in ChromaDB with metadata
   ```

2. **Query Processing** (per question)
   ```
   Question: "What is the encoder?"
   → Convert to embedding: [0.25, -0.42, 0.65, ...]
   → Similarity search in ChromaDB
   → Find 6 most relevant chunks (MMR)
   → Chunks contain: "encoder", "neural network", "transformer"
   ```

3. **Answer Generation**
   ```
   Prompt to Gemini:
   Context: [Retrieved 6 chunks about encoders]
   Question: "What is the encoder?"
   
   Gemini Response:
   "The encoder is a neural network component that processes
   input sequences and converts them into a representation..."
   
   Sources: Page 3, Page 5, Page 7
   ```

---

## 📁 Project Structure

```
VBRAG/
│
├── main1.py                # Streamlit web interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── .env                   # API keys (not in repo)
│
├── article.pdf            # Sample PDF (optional)
├── temp.pdf               # Temporary files (ignored)
│
└── chroma_db_*/           # Vector store directories (ignored)
    ├── chroma.sqlite3
    └── ...
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "GOOGLE_API_KEY is missing"**
- **Solution**: Create `.env` file with `GOOGLE_API_KEY=your_key`
- **Check**: File is in project root, not in subdirectories

**2. "PDF file not found"**
- **Solution**: Upload PDF file through Streamlit interface
- **Note**: The web interface handles file uploads automatically

**3. "No valid text chunks could be extracted"**
- **Cause**: PDF might be image-based (scanned)
- **Solution**: Use OCR tools to extract text first

**4. Import errors**
- **Solution**: 
  ```bash
  pip install -r requirements.txt --upgrade
  ```

**5. ChromaDB errors**
- **Solution**: Delete `chroma_db_*/` directories and reprocess
- **Cause**: Corrupted vector store

**6. Slow processing**
- **First time**: Normal (generating embeddings)
- **Subsequent**: Should be fast (cached)
- **Optimization**: Use GPU for embeddings (if available)

---

## 🚀 Future Enhancements

### Planned Features

1. **Multi-Document Support**
   - Process multiple PDFs in one session
   - Cross-document question answering

2. **Advanced Retrieval**
   - Hybrid search (keyword + semantic)
   - Re-ranking with cross-encoders
   - Similarity score thresholds

3. **Conversation History**
   - Follow-up questions
   - Context-aware responses
   - Chat interface

4. **Additional Formats**
   - DOCX, TXT, Markdown support
   - Web scraping
   - Database integration

5. **Performance Optimizations**
   - GPU acceleration for embeddings
   - Batch processing
   - Async operations

6. **UI Improvements**
   - Answer confidence scores
   - Visual similarity graphs
   - Export conversations

---

## 📚 Additional Resources

### Learn More About RAG

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HuggingFace Embeddings](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/docs)

### Related Concepts

- **Vector Databases**: ChromaDB, Pinecone, Weaviate
- **Embedding Models**: OpenAI, Cohere, Sentence Transformers
- **RAG Patterns**: Parent-Child Chunking, Metadata Filtering, Query Expansion

---

## 📄 License

This project is open source and available for educational and commercial use.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional document formats
- Better chunking strategies
- UI/UX enhancements
- Performance optimizations

---

## 📧 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review code comments
3. Check library documentation

---

**Built with ❤️ using LangChain, ChromaDB, HuggingFace, and Google Gemini**
#   I n t e l l i g e n t - D o c u m e n t - Q - A - S y s t e m -  
 