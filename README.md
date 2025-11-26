# VBRAG - PDF Q&A with RAG

A simple RAG (Retrieval Augmented Generation) application that answers questions from PDF documents using Google Gemini AI.

## Features

- 📄 Upload and process PDF documents
- 🔍 Semantic search using vector embeddings
- 💬 Ask questions and get answers with source citations
- 🚀 Fast processing with persistent vector storage
- 📊 Web interface built with Streamlit

## How It Works

1. **Upload PDF** → Document is split into chunks
2. **Generate Embeddings** → Text converted to vectors using HuggingFace
3. **Store Vectors** → Saved in ChromaDB for fast search
4. **Ask Questions** → System finds relevant chunks and generates answers using Gemini

## Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key ([Get one here](https://aistudio.google.com/))

### Setup

1. **Clone the repository**
   ```bash
   cd VBRAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run main1.py
```

Then:
1. Open browser at `http://localhost:8501`
2. Upload a PDF file
3. Wait for processing (first time takes a few seconds)
4. Ask questions about the document
5. View answers with source page references

## Technology Stack

- **LangChain** - RAG pipeline orchestration
- **ChromaDB** - Vector database for embeddings
- **HuggingFace** - Text embeddings (sentence-transformers)
- **Google Gemini** - LLM for answer generation
- **Streamlit** - Web interface

## Project Structure

```
VBRAG/
├── main1.py              # Streamlit application
├── requirements.txt      # Dependencies
├── README.md            # This file
├── .env                 # API keys (create this)
└── chroma_db_*/          # Vector stores (auto-generated)
```

## Troubleshooting

**API Key Error**
- Make sure `.env` file exists with `GOOGLE_API_KEY=your_key`

**Import Errors**
- Run: `pip install -r requirements.txt --upgrade`

**Slow Processing**
- First time is normal (generating embeddings)
- Subsequent loads are fast (cached)

**PDF Not Loading**
- Ensure PDF has extractable text (not just images)
- Try a different PDF file

## Configuration

Key parameters in `main1.py`:
- `chunk_size=1000` - Size of text chunks
- `chunk_overlap=200` - Overlap between chunks
- `k=6` - Number of chunks to retrieve
- `temperature=0.3` - LLM creativity (lower = more focused)

## License

Open source - feel free to use and modify.

---

**Built with LangChain, ChromaDB, HuggingFace, and Google Gemini**
