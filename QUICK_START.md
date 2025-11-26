# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup (2 minutes)

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo GOOGLE_API_KEY=your_key_here > .env
```

### 2. Run Web Application (1 minute)

```bash
streamlit run main1.py

# Browser opens automatically
# Upload PDF and ask questions
```

---

## 📝 Key Concepts

### RAG Pipeline

```
PDF → Chunks → Embeddings → Vector Store → Retrieve → Answer
```

### Main Components

1. **Document Loader**: Extracts text from PDF
2. **Text Splitter**: Creates chunks (1000 chars, 200 overlap)
3. **Embeddings**: Converts text to vectors (384-dim)
4. **Vector Store**: Stores and searches vectors (ChromaDB)
5. **Retriever**: Finds relevant chunks (MMR algorithm)
6. **LLM**: Generates answers (Gemini 2.0 Flash)

---

## 🔧 Configuration

### Change Chunk Size

**In `main1.py`:**
```python
CHUNK_SIZE = 1500  # Default: 1000
CHUNK_OVERLAP = 300  # Default: 200
```

### Change Retrieval Count

```python
NUM_RETRIEVED_CHUNKS = 8  # Default: 6
```

### Change LLM Model

```python
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Different model
    temperature=0.5  # More creative (default: 0.3)
)
```

---

## 📊 Understanding Output

### CLI Output

```
Answer: [Generated answer text]

Source Information: Found 6 relevant document chunks

Source 1 (Page 3):
[Chunk text preview...]
```

### Web UI Output

- **Answer Section**: Main generated answer
- **Source References**: Expandable sections with:
  - Page numbers
  - File names
  - Full chunk text

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| API key error | Check `.env` file exists and has correct key |
| PDF not found | Ensure file is in project directory |
| Slow processing | Normal on first run (generating embeddings) |
| Import errors | Run `pip install -r requirements.txt` |

---

## 📚 Next Steps

1. Read [README.md](README.md) for full documentation
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Experiment with different chunk sizes
4. Try different PDFs
5. Modify the prompt template for different use cases

---

## 💡 Tips

- **First run is slow**: Embeddings are generated (5-10s)
- **Subsequent runs are fast**: Vector store is cached
- **Better chunks = better answers**: Tune chunk size for your documents
- **MMR helps diversity**: Prevents redundant information
- **Temperature matters**: Lower = more focused, Higher = more creative

---

**Happy querying! 🎉**

