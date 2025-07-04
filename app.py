import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import pdfplumber
import os
import pickle
import time

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # or "BAAI/bge-small-en"
LLM_MODEL_NAME = "distilgpt2"  # Changed to open, small model
CHUNK_SIZE = 200  # words per chunk

# --- 1. Document Preparation & Ingestion ---

def clean_text(text):
    # Remove headers, footers, and extra whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    # Sentence-aware chunking
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        words = sent.split()
        if current_length + len(words) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sent)
        current_length += len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def embed_chunks(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- 2. Model Loading ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, do_sample=True)

# --- 3. RAG Pipeline ---

def retrieve(query, embedding_model, faiss_index, chunks, top_k=3):
    query_emb = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(query_emb).astype('float32'), top_k)
    return [chunks[i] for i in I[0]], I[0]

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question factually and only based on the context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    return prompt

# --- 4. Streamlit UI ---

def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("RAG Chatbot")

    # Sidebar
    st.sidebar.title("Info")
    st.sidebar.write(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")
    st.sidebar.write(f"**LLM Model:** {LLM_MODEL_NAME}")
    if "chunks" in st.session_state:
        st.sidebar.write(f"**# Chunks:** {len(st.session_state['chunks'])}")

    # Option to load precomputed FAISS index and chunks
    use_precomputed = st.sidebar.checkbox("Use precomputed FAISS index & chunks", value=os.path.exists("faiss_index.bin") and os.path.exists("chunks.pkl"))

    if use_precomputed and os.path.exists("faiss_index.bin") and os.path.exists("chunks.pkl"):
        if "faiss_index" not in st.session_state:
            with open("chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            faiss_index = faiss.read_index("faiss_index.bin")
            embedding_model = load_embedding_model()
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = faiss_index
            st.session_state["embedding_model"] = embedding_model
        uploaded_file = None  # Disable upload if using precomputed
    else:
        # Upload document
        uploaded_file = st.file_uploader("Upload your document (.txt or .pdf)", type=["txt", "pdf"])
        if uploaded_file and "faiss_index" not in st.session_state:
            if uploaded_file.name.lower().endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            else:
                text = uploaded_file.read().decode("utf-8")
            text = clean_text(text)
            chunks = chunk_text(text)
            embedding_model = load_embedding_model()
            embeddings = embed_chunks(chunks, embedding_model)
            faiss_index = build_faiss_index(embeddings)
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = faiss_index
            st.session_state["embedding_model"] = embedding_model

    # Chat
    if "faiss_index" in st.session_state:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("Ask a question about the document:")
        if st.button("Send") and user_input:
            retrieved, indices = retrieve(
                user_input,
                st.session_state["embedding_model"],
                st.session_state["faiss_index"],
                st.session_state["chunks"],
                top_k=3
            )
            prompt = build_prompt(user_input, retrieved)
            llm = load_llm()
            with st.spinner("Generating answer..."):
                response = llm(prompt)[0]["generated_text"].split("Answer:")[-1].strip()
                response_text = ""
                response_placeholder = st.empty()
                for sentence in response.split('. '):
                    response_text += sentence + '. '
                    response_placeholder.markdown(response_text)
                    time.sleep(0.05)  # Simulate streaming
            st.session_state["chat_history"].append(
                {"user": user_input, "bot": response_text, "sources": [st.session_state["chunks"][i] for i in indices]}
            )

        # Display chat
        for entry in st.session_state.get("chat_history", []):
            st.markdown(f"**You:** {entry['user']}")
            st.markdown(f"**Bot:** {entry['bot']}")
            with st.expander("Show source passages"):
                for src in entry["sources"]:
                    st.write(src)
            st.markdown("---")

        if st.button("Clear Chat"):
            st.session_state["chat_history"] = []

if __name__ == "__main__":
    main() 