import pdfplumber
import re
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

PDF_PATH = "AI Training Document.pdf"
CHUNK_SIZE = 200  # words per chunk
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 2. Clean text
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 3. Chunk text (sentence-aware)
def chunk_text(text, chunk_size=CHUNK_SIZE):
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

# 4. Generate embeddings
def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# 5. Store in FAISS
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    print("Cleaning text...")
    text = clean_text(text)
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Generating embeddings...")
    embeddings = embed_chunks(chunks, model)
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"Saving FAISS index to {FAISS_INDEX_PATH} and chunks to {CHUNKS_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("Done.") 