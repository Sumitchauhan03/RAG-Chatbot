# RAG Chatbot

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit, FAISS, and transformer-based models. It allows users to upload documents (PDF or TXT), processes them into semantic chunks, builds a vector index, and enables question-answering over the document with streaming responses from a language model.

---

## Architecture & Flow


- **Document Ingestion:** PDF/TXT files are uploaded and cleaned.
- **Chunking:** Text is split into sentence-aware chunks (~200 words).
- **Embedding:** Chunks are embedded using a SentenceTransformer.
- **Indexing:** Embeddings are stored in a FAISS index for fast retrieval.
- **Retrieval:** User queries are embedded and matched to top-K chunks.
- **Generation:** Retrieved context is passed to a language model (LLM) for answer generation, streamed to the UI.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Preprocessing & Embedding Creation

You can preprocess a document and build the FAISS index in two ways:

### 1. **Automatic (via UI)**
- Upload a `.pdf` or `.txt` file in the Streamlit app. The app will extract, clean, chunk, embed, and index the document on the fly.

### 2. **Manual (Precompute for Large Docs)**
- Use the provided script:
  ```bash
  python process_pdf_to_faiss.py
  ```
- This will:
  - Extract and clean text from `AI Training Document.pdf`
  - Chunk the text
  - Generate embeddings using `all-MiniLM-L6-v2`
  - Build and save a FAISS index (`faiss_index.bin`) and chunk list (`chunks.pkl`)
- In the Streamlit app, check "Use precomputed FAISS index & chunks" to load these files for faster startup.

---

## Model & Embedding Choices

- **Embedding Model:** `all-MiniLM-L6-v2` (from `sentence-transformers`)
  - Chosen for its balance of speed and semantic accuracy.
- **LLM Model:** `distilgpt2` (from HuggingFace Transformers)
  - Lightweight, open-source, suitable for local inference and streaming.
- **Vector Store:** FAISS (FlatL2 index)
  - Efficient similarity search for dense embeddings.

You can change models by editing the `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME` variables in the code.

---

## Running the Chatbot (with Streaming)

1. **Start the Streamlit app:**
   ```bash
   streamlit run rag_chatbot.py
   ```
2. **Upload a document** (or use precomputed index).
3. **Ask questions** in the chat box. Answers will stream in real-time.
4. **View source passages** for each answer by expanding the "Show source passages" section.

---

## Sample Queries

- "What is the main objective of the document?"
- "Summarize the key points from section 2."
- "List the steps involved in the training process."
- "Who is the intended audience for this document?"

**Sample Output:**
```
You: What is the main objective of the document?
Bot: The main objective of the document is to provide an overview of the AI training process, including methodologies, best practices, and evaluation metrics, as described in the provided context.
```

---

## Demo

https://drive.google.com/file/d/1Iazbi8IxASH4Fmrk_ENY-YQAMCKJ3v_4/view?usp=sharing

---

## Notes
- For large documents, precomputing the FAISS index is recommended.
- All processing is local; no data is sent to external servers.
- You can adjust chunk size, top-K retrieval, and model choices in the code.

---

## License
MIT 
