import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile
import hashlib

# Load .env and API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key or not api_key.strip():
    st.error("GOOGLE_API_KEY is missing. Add it to your environment or .env file.")
    st.stop()

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key, temperature=0.3)

# Streamlit UI
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("dY: Ask Questions from Your PDF (Gemini + LangChain)")

# Custom Prompt Template - Improved for better RAG performance
custom_prompt_template = """Use the following pieces of context from the document to answer the question at the end. 
If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.
Keep the answer concise but comprehensive. If the context contains relevant information, use it to provide a detailed answer.
Cite specific information from the context when possible.

Context: {context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

@st.cache_resource
def process_pdf(file_path, file_name):
    """Process PDF and create vector store with metadata tracking"""
    # Load and split PDF
    pdfloader = PyPDFLoader(file_path)
    loaded_pdf_doc = pdfloader.load()
    
    if not loaded_pdf_doc:
        raise ValueError("PDF file is empty or could not be loaded")
    
    # Add metadata to documents (file name, page numbers)
    for i, doc in enumerate(loaded_pdf_doc):
        doc.metadata['source_file'] = file_name
        doc.metadata['page'] = doc.metadata.get('page', i)
    
    # Optimized Chunking: Larger chunks for better context retention
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=['\n\n', '\n', '. ', ' ', '']  # Better separation strategy
    )
    chunks = splitter.split_documents(loaded_pdf_doc)
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    if not chunks:
        raise ValueError("No valid text chunks could be extracted from the PDF")
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create persistent vector store (based on file hash for caching)
    file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
    persist_dir = f"./chroma_db_{file_hash}"
    
    # Check if vector store already exists
    if os.path.exists(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        st.info(f"Using cached vector store for this document")
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    return vector_store, len(chunks), len(loaded_pdf_doc)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Button to clear session state and process a new PDF
if 'vector_store' in st.session_state:
    if st.button("Clear and Upload New PDF"):
        st.session_state.clear()
        st.rerun()

if uploaded_file is not None:
    # Validate file type
    if uploaded_file.type != "application/pdf":
        st.error("Please upload a valid PDF file.")
        st.stop()
    
    # Initialize session state for vector store and chain
    if 'vector_store' not in st.session_state or 'rag_chain' not in st.session_state:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        with st.spinner("Processing PDF..."):
            try:
                # Process PDF with metadata tracking
                vector_store, num_chunks, num_pages = process_pdf(temp_file_path, uploaded_file.name)
                
                # Display processing statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages Processed", num_pages)
                with col2:
                    st.metric("Text Chunks Created", num_chunks)
                with col3:
                    st.metric("File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
                
                # Smarter Retrieval: MMR to find diverse, relevant chunks
                retriever = vector_store.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance for diversity
                    search_kwargs={
                        "k": 6,  # Number of chunks to retrieve
                        "fetch_k": 20,  # Fetch more candidates for MMR
                        "lambda_mult": 0.5  # Balance between relevance (1.0) and diversity (0.0)
                    }
                )

                # Build RAG chain with custom prompt
                rag_chain = RetrievalQA.from_chain_type(
                    llm=model,
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )

                # Store in session state to avoid reprocessing
                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = rag_chain
                st.session_state.file_name = uploaded_file.name
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                st.success("PDF processed successfully! You can now ask questions.")

            except Exception as e:
                st.error(f"An error occurred while processing PDF: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    else:
        st.info("Using previously processed PDF. Upload a new file to process a different PDF.")

    # Input box for user query
    st.markdown("---")
    query = st.text_input("Ask a question from the PDF:", "", key="query_input")
    
    if query and 'rag_chain' in st.session_state:
        with st.spinner("Searching document and generating answer..."):
            try:
                # Use .invoke() to get answer from RAG chain
                response = st.session_state.rag_chain.invoke({"query": query})
                answer = response['result']
                sources = response.get('source_documents', [])

                # Display answer
                st.markdown("### 💡 Answer:")
                st.write(answer)
                
                # Display source information with metadata
                if sources:
                    st.markdown("### 📚 Source References:")
                    st.caption(f"Found {len(sources)} relevant sections from the document")
                    
                    for i, doc in enumerate(sources, 1):
                        page_num = doc.metadata.get('page', 'N/A')
                        source_file = doc.metadata.get('source_file', 'Unknown')
                        
                        with st.expander(f"📄 Source {i} - Page {page_num} ({source_file})"):
                            st.write(doc.page_content)
                            # Show metadata
                            with st.container():
                                st.caption(f"**Metadata:** Page {page_num} | File: {source_file}")
                else:
                    st.info("No source documents found for this question.")
                    
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    elif query:
        st.warning("Please wait for the PDF to finish processing before asking questions.")
