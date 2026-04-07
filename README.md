# Multimodal Retrieval-Augmented Generation (RAG) System

A complete end-to-end **Multimodal RAG pipeline** that processes PDF documents containing text, tables, and images, enabling intelligent question answering using both textual and visual queries.

This system integrates **semantic retrieval with large language models (LLMs)** to generate accurate, context-aware responses through a ChatGPT-like interface.

---

## Overview

This project implements a production-level Retrieval-Augmented Generation system capable of:

* Extracting structured and unstructured data from PDFs
* Understanding both textual and visual content
* Performing semantic search using vector embeddings
* Generating grounded answers using retrieved context
* Supporting advanced prompting strategies
* Evaluating retrieval and generation quality

---

## Key Features

### Multimodal Retrieval

* Text-based queries
* Image-based queries using CLIP embeddings
* Cross-modal understanding (text ↔ image)

### End-to-End RAG Pipeline

* PDF parsing (text + images)
* Chunking with metadata
* Embedding generation
* FAISS vector indexing
* Top-K semantic retrieval
* Context-aware answer generation

### Prompt Engineering

* Zero-shot prompting
* Few-shot prompting
* Chain-of-Thought (CoT) reasoning

### Evaluation Metrics

* BLEU
* ROUGE (ROUGE-1, ROUGE-L)
* Cosine similarity
* Precision@K, Recall@K, MAP
* Query latency

### Visualization

* t-SNE embedding space visualization
* Semantic clustering of multimodal data

---

## System Architecture

```
PDFs → Extraction → Chunking → Embeddings → FAISS Index
                                      ↓
User Query → Embedding → Retrieval → Context → LLM → Answer
```

---

## Tech Stack

### Core

* Python
* Streamlit

### NLP & Vision

* SentenceTransformers (text embeddings)
* CLIP (image embeddings)
* Tesseract OCR

### Retrieval

* FAISS (vector database)

### Language Model

* OpenAI GPT (gpt-4o-mini)

### Evaluation & Visualization

* Scikit-learn
* BLEU / ROUGE
* t-SNE / PCA

---

## Project Structure

```
.
├── app.py              # Streamlit interface
├── rag_engine.py       # Core RAG pipeline
├── models.py           # Embedding, OCR, and LLM modules
├── utils.py            # Metrics and visualization
├── data/               # Input PDF documents
├── db/                 # FAISS index (generated locally)
├── extracted_images/   # Extracted images from PDFs
```

---

## How It Works

### 1. Data Extraction

* PDFs are parsed using PyMuPDF
* Text and images are extracted per page
* OCR is applied to images for semantic understanding

### 2. Chunking

* Text is split into smaller chunks
* Each chunk is stored with metadata:

  * Source document
  * Page number
  * Content type

### 3. Embedding Generation

* Text → SentenceTransformer → projected to 768 dimensions
* Images → CLIP → projected to 768 dimensions

### 4. Vector Storage

* FAISS IndexFlatL2 is used
* Enables fast similarity search

### 5. Retrieval

* Query is embedded (text or image)
* Top-K relevant chunks are retrieved

### 6. Answer Generation

* Retrieved context is passed to LLM
* Final answer generated using prompt strategies

---

## User Interface

Built using Streamlit with three main modules:

### Text Query

* Enter natural language queries
* View ranked retrieved chunks
* Get final generated answer

### Image Query

* Upload image
* Ask questions about the image
* Retrieve relevant visual and textual context

### Evaluation

* Compare generated answer with reference answer
* Metrics displayed:

  * BLEU
  * ROUGE
  * Precision@K / Recall@K
  * MAP
  * Latency

### Visualization

* t-SNE plot of embedding space
* Demonstrates semantic clustering

---

## Results

* Precision@3: 1.0
* Recall@3: 1.0
* ROUGE-L: ~0.72
* Cosine Similarity: ~0.78
* Latency: ~0.007 seconds

Embedding visualization shows clear clustering, validating retrieval effectiveness.

---

## Setup Instructions

### 1. Clone Repository

```
git clone https://github.com/your-username/multimodal-rag-system.git
cd multimodal-rag-system
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key
```

### 4. Run Application

```
streamlit run app.py
```

---

## Usage

1. Click **Build Vector Index**
2. Enter a query (text or image)
3. Select prompting strategy
4. View retrieved results and generated answer
5. Optionally evaluate performance

---

## Key Highlights

* Complete end-to-end AI system implementation
* Multimodal understanding (text + images)
* Strong use of vector databases and embeddings
* Integration of LLM with retrieval grounding
* Includes evaluation and visualization components

---
## Results

Main Streamlit interface for RAG system.
<img width="1205" height="430" alt="image" src="https://github.com/user-attachments/assets/ee7a4870-1069-4240-8b74-043060bdc458" />

Example image query result showing retrieved chunks and LLM-generated answer.
<img width="737" height="372" alt="image" src="https://github.com/user-attachments/assets/ed46cf7c-24b3-45a8-8923-62bf6ac3d228" />

Text query retrieval example showing ranked chunks and LLM answer.
<img width="1230" height="606" alt="image" src="https://github.com/user-attachments/assets/954bcbb8-2296-479d-bfa3-1c2b8907d589" />

Image query interface.
<img width="733" height="370" alt="image" src="https://github.com/user-attachments/assets/1e6ba04c-1407-4e57-8cf9-a776dcbac881" />

