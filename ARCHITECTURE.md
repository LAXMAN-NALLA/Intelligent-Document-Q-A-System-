# Technical Architecture Documentation

## Deep Dive into VBRAG Implementation

This document provides detailed technical information about how VBRAG is implemented, including design decisions, algorithms, and performance considerations.

---

## Table of Contents

1. [System Design](#system-design)
2. [Data Structures](#data-structures)
3. [Algorithms](#algorithms)
4. [Performance Considerations](#performance-considerations)
5. [Design Decisions](#design-decisions)
6. [Code Flow Diagrams](#code-flow-diagrams)

---

## System Design

### Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      User Layer                              │
├──────────────────────────────────────────────────────────────┤
│  CLI Interface (main.py)    │    Web UI (main1.py)          │
│  - File validation          │    - File upload              │
│  - Interactive loop         │    - Session management       │
│  - Error handling           │    - Caching                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
├──────────────────────────────────────────────────────────────┤
│  Document Loader                                             │
│  ├─ PyPDFLoader: Extracts text + metadata                   │
│  └─ Output: List[Document] with page numbers                │
│                                                              │
│  Text Splitter                                               │
│  ├─ RecursiveCharacterTextSplitter                           │
│  ├─ Strategy: Hierarchical separators                        │
│  ├─ Chunk size: 1000 chars                                  │
│  └─ Overlap: 200 chars                                      │
│                                                              │
│  Embedding Generator                                         │
│  ├─ Model: all-MiniLM-L6-v2                                 │
│  ├─ Dimensions: 384                                         │
│  ├─ Batch processing: Yes                                   │
│  └─ Output: NumPy arrays                                    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Storage Layer                              │
├──────────────────────────────────────────────────────────────┤
│  Vector Database (ChromaDB)                                  │
│  ├─ Storage: SQLite + HNSW index                             │
│  ├─ Persistence: Disk-based                                  │
│  ├─ Metadata: Page numbers, file names                      │
│  └─ Index: Approximate Nearest Neighbor (ANN)                │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Retrieval Layer                            │
├──────────────────────────────────────────────────────────────┤
│  Retriever                                                   │
│  ├─ Algorithm: Maximum Marginal Relevance (MMR)              │
│  ├─ Search: Cosine similarity                                │
│  ├─ Parameters: k=6, fetch_k=20, lambda=0.5                │
│  └─ Output: Ranked document chunks                          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Generation Layer                           │
├──────────────────────────────────────────────────────────────┤
│  RAG Chain (RetrievalQA)                                     │
│  ├─ Input: Question + Retrieved chunks                       │
│  ├─ Prompt: Custom template                                 │
│  ├─ LLM: Gemini 2.0 Flash                                    │
│  ├─ Temperature: 0.3                                        │
│  └─ Output: Answer + Source documents                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Document Object

```python
Document(
    page_content: str,           # The actual text
    metadata: {
        'page': int,            # Page number (0-indexed)
        'source': str,          # File path
        'source_file': str      # File name (in main1.py)
    }
)
```

### Chunk Object

After splitting, chunks inherit Document structure:
```python
Chunk(
    page_content: str,          # Text chunk (1000 chars)
    metadata: {
        'page': int,           # Original page number
        'source': str,         # Source file
        'chunk_index': int     # Position in document
    }
)
```

### Vector Store Entry

```python
{
    'id': str,                  # Unique identifier
    'embedding': [float],       # 384-dimensional vector
    'document': str,            # Text content
    'metadata': {
        'page': int,
        'source_file': str
    }
}
```

### RAG Response

```python
{
    'result': str,              # Generated answer
    'source_documents': [       # Retrieved chunks
        Document(...),
        Document(...),
        ...
    ]
}
```

---

## Algorithms

### 1. Text Chunking Algorithm

**RecursiveCharacterTextSplitter** uses hierarchical splitting:

```python
def split_text(text, separators=['\n\n', '\n', '. ', ' ', '']):
    chunks = []
    current_chunk = ""
    
    for separator in separators:
        if len(current_chunk) + len(separator) <= chunk_size:
            # Try to split by this separator
            parts = text.split(separator)
            for part in parts:
                if len(current_chunk + part) <= chunk_size:
                    current_chunk += part + separator
                else:
                    # Save current chunk, start new one
                    chunks.append(current_chunk)
                    current_chunk = part + separator
        else:
            # Move to next separator
            continue
    
    return chunks
```

**Overlap Strategy:**
- Last `overlap_size` characters of chunk N
- First `overlap_size` characters of chunk N+1
- Ensures no information loss at boundaries

### 2. Embedding Generation

**HuggingFace Sentence Transformer Pipeline:**

```python
def generate_embedding(text):
    # 1. Tokenization
    tokens = tokenizer(text, padding=True, truncation=True, max_length=512)
    
    # 2. Model forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    
    # 3. Mean pooling (average token embeddings)
    embeddings = mean_pooling(outputs, tokens['attention_mask'])
    
    # 4. Normalization
    embeddings = normalize(embeddings)
    
    return embeddings  # Shape: (1, 384)
```

**Why Mean Pooling?**
- Averages all token embeddings
- Captures overall sentence meaning
- Standard for sentence transformers

### 3. Similarity Search (ChromaDB)

**HNSW (Hierarchical Navigable Small World) Algorithm:**

```
1. Build graph structure
   - Each vector is a node
   - Connect to nearest neighbors
   - Multiple layers (hierarchical)

2. Search process
   - Start at top layer
   - Navigate to nearest neighbor
   - Move to lower layer
   - Repeat until bottom layer
   - Return k nearest neighbors
```

**Time Complexity:**
- Build: O(N log N)
- Search: O(log N) (approximate)
- Much faster than brute force O(N)

**Distance Metric:**
- Cosine Similarity: `cos(θ) = (A · B) / (||A|| ||B||)`
- Range: [-1, 1]
- 1 = identical, 0 = orthogonal, -1 = opposite

### 4. Maximum Marginal Relevance (MMR)

**Algorithm:**

```python
def mmr_selection(query_embedding, candidate_chunks, k, lambda_mult):
    selected = []
    candidates = candidate_chunks.copy()
    
    # First: Select most relevant
    first = max(candidates, key=lambda x: similarity(query_embedding, x))
    selected.append(first)
    candidates.remove(first)
    
    # Subsequent: Balance relevance and diversity
    while len(selected) < k:
        best_score = -inf
        best_chunk = None
        
        for candidate in candidates:
            # Relevance to query
            relevance = similarity(query_embedding, candidate)
            
            # Diversity from already selected
            max_similarity = max([
                similarity(candidate, s) 
                for s in selected
            ])
            
            # MMR score
            score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            
            if score > best_score:
                best_score = score
                best_chunk = candidate
        
        selected.append(best_chunk)
        candidates.remove(best_chunk)
    
    return selected
```

**Lambda Parameter:**
- `lambda_mult = 0.0`: Pure diversity (ignore query relevance)
- `lambda_mult = 0.5`: Balanced (our choice)
- `lambda_mult = 1.0`: Pure relevance (no diversity)

### 5. RAG Chain Execution

**RetrievalQA Chain Flow:**

```python
def rag_chain_invoke(query):
    # Step 1: Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)
    
    # Step 2: Format context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step 3: Format prompt
    prompt = PROMPT.format(context=context, question=query)
    
    # Step 4: Generate answer
    answer = llm.invoke(prompt)
    
    # Step 5: Return with sources
    return {
        'result': answer,
        'source_documents': docs
    }
```

---

## Performance Considerations

### Processing Time Breakdown

**First-time PDF Processing:**
- PDF Loading: ~0.1-1s (depends on size)
- Text Splitting: ~0.01s per page
- Embedding Generation: ~0.1s per chunk (CPU)
  - 45 chunks × 0.1s = ~4.5s
- Vector Store Creation: ~0.5-2s
- **Total: ~5-8s for 10-page document**

**Subsequent Queries:**
- Query Embedding: ~0.1s
- Similarity Search: ~0.01-0.05s (HNSW)
- LLM Generation: ~1-3s (API call)
- **Total: ~1-3s per question**

### Optimization Strategies

**1. Caching (Implemented)**
- Vector stores persisted to disk
- Same file = instant load
- Session state caching in Streamlit

**2. Batch Processing**
- Embeddings generated in batches
- Reduces overhead

**3. Approximate Search**
- HNSW provides fast approximate search
- Trade-off: 99% accuracy for 100x speed

**4. Chunk Size Tuning**
- Larger chunks = fewer embeddings = faster
- But: Less precise retrieval
- Our choice (1000) balances both

### Memory Usage

**Per Document:**
- Text: ~1-5 MB (depends on PDF)
- Embeddings: 45 chunks × 384 dims × 4 bytes = ~69 KB
- Metadata: Negligible
- **Total: ~1-5 MB per document**

**Vector Store:**
- ChromaDB overhead: ~10-20% of data size
- Index (HNSW): ~20-30% of data size
- **Total overhead: ~30-50%**

---

## Design Decisions

### Why ChromaDB over Alternatives?

| Feature | ChromaDB | Pinecone | Weaviate | FAISS |
|---------|----------|----------|----------|-------|
| Setup | Embedded | Cloud | Server | Library |
| Cost | Free | Paid | Free/Paid | Free |
| Persistence | Yes | Yes | Yes | Manual |
| Metadata | Yes | Yes | Yes | No |
| Our Choice | ✅ | ❌ | ❌ | ❌ |

**Decision**: ChromaDB provides best balance of features, ease of use, and cost.

### Why HuggingFace Embeddings over OpenAI?

| Feature | HuggingFace | OpenAI |
|---------|-------------|--------|
| Cost | Free | $0.0001/1K tokens |
| Speed | Fast (local) | API latency |
| Privacy | Local | Cloud |
| Quality | Good | Excellent |
| Our Choice | ✅ | ❌ |

**Decision**: Free, fast, private, and good enough quality.

### Why MMR over Simple Similarity?

**Simple Similarity:**
- Fast
- May return redundant chunks
- Example: 6 chunks all saying the same thing

**MMR:**
- Slightly slower
- Returns diverse chunks
- Example: 6 chunks covering different aspects

**Decision**: MMR provides better answer quality with minimal performance cost.

### Why Chunk Size 1000?

**Too Small (300):**
- ✅ Precise retrieval
- ❌ Lost context
- ❌ More chunks = slower

**Too Large (2000):**
- ✅ More context
- ❌ Irrelevant information
- ❌ Less precise

**Our Choice (1000):**
- ✅ Balanced
- ✅ Good context retention
- ✅ Reasonable precision

---

## Code Flow Diagrams

### Document Processing Flow

```
PDF File
    │
    ▼
PyPDFLoader.load()
    │
    ▼
List[Document] (with page numbers)
    │
    ▼
RecursiveCharacterTextSplitter.split_documents()
    │
    ▼
List[Chunk] (1000 chars each, 200 overlap)
    │
    ▼
HuggingFaceEmbeddings.embed_documents()
    │
    ▼
List[Vectors] (384-dim each)
    │
    ▼
ChromaDB.from_documents()
    │
    ▼
Persistent Vector Store
```

### Query Processing Flow

```
User Question
    │
    ▼
HuggingFaceEmbeddings.embed_query()
    │
    ▼
Query Vector (384-dim)
    │
    ▼
ChromaDB.similarity_search_with_score()
    │
    ▼
Top 20 Candidates (with scores)
    │
    ▼
MMR Algorithm
    │
    ▼
Top 6 Diverse Chunks
    │
    ▼
Prompt Template.format()
    │
    ▼
Formatted Prompt (Context + Question)
    │
    ▼
Gemini LLM.invoke()
    │
    ▼
Answer + Source Documents
```

### Session Management (Streamlit)

```
User Uploads PDF
    │
    ▼
Check session_state['vector_store']
    │
    ├─ Exists → Use cached
    │
    └─ Not exists → Process PDF
            │
            ▼
        Save to session_state
            │
            ▼
        Persist to disk (ChromaDB)
            │
            ▼
        Ready for queries
```

---

## Error Handling Strategy

### Validation Points

1. **File Validation**
   - File exists
   - File is PDF
   - File is not empty

2. **Processing Validation**
   - Text extraction successful
   - Chunks created
   - Embeddings generated

3. **Query Validation**
   - Question not empty
   - Vector store exists
   - Retrieval successful

### Error Recovery

- **File errors**: Clear error message, suggest alternatives
- **Processing errors**: Show traceback, allow retry
- **Query errors**: Graceful degradation, show partial results

---

## Security Considerations

1. **API Key Management**
   - Stored in `.env` (not in code)
   - Excluded from version control
   - Never logged or displayed

2. **File Handling**
   - Temporary files deleted after processing
   - No persistent storage of uploaded files
   - File type validation

3. **Data Privacy**
   - Embeddings generated locally
   - Only question + context sent to API
   - No document content sent to external services (except LLM)

---

## Scalability Considerations

### Current Limitations

- **Single document processing**: One PDF at a time
- **In-memory processing**: Large PDFs may be slow
- **Synchronous operations**: No async/parallel processing

### Future Improvements

1. **Batch Processing**: Process multiple PDFs
2. **Async Operations**: Parallel embedding generation
3. **Distributed Storage**: Use cloud vector database
4. **Caching Layer**: Redis for frequently accessed documents

---

## Testing Strategy

### Unit Tests (Recommended)

- Document loading
- Text splitting
- Embedding generation
- Similarity search
- MMR algorithm

### Integration Tests (Recommended)

- End-to-end RAG pipeline
- Error handling
- Session management

### Performance Tests (Recommended)

- Processing time benchmarks
- Memory usage profiling
- Query latency measurements

---

**This architecture document provides the technical foundation for understanding and extending VBRAG.**

