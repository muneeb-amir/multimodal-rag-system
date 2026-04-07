# rag_engine.py

import os
import time
import fitz
import faiss
import pickle
import numpy as np

from models import OCRModel, TextEmbedder, ImageEmbedder, LLM
from utils import chunk_text


BASE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------
# PDF EXTRACTION
# ----------------------------
def extract_pdf(pdf_path, img_dir=None):
    if img_dir is None:
        img_dir = os.path.join(BASE, "extracted_images")
    os.makedirs(img_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    text_items, image_items = [], []

    for page_no, page in enumerate(doc):
        # TEXT
        text = page.get_text()
        if text.strip():
            text_items.append({
                "type": "text",
                "content": text,
                "page": page_no + 1,
                "source": os.path.basename(pdf_path)
            })

        # IMAGES
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_path = os.path.join(
                img_dir,
                f"{os.path.basename(pdf_path)}_{page_no}_{img_index}.png"
            )

            try:
                if pix.colorspace is None or pix.colorspace.n >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(img_path)
            except:
                continue

            image_items.append({
                "type": "image",
                "path": img_path,
                "page": page_no + 1,
                "source": os.path.basename(pdf_path)
            })

    return text_items, image_items


# ----------------------------
# CHUNKING
# ----------------------------
def prepare_chunks(text_items, image_items):
    ocr = OCRModel()
    chunks = []
    cid = 0

    # TEXT CHUNKS
    for item in text_items:
        for ch in chunk_text(item["content"]):
            chunks.append({
                "id": cid,
                "type": "text",
                "content": ch,
                "metadata": item
            })
            cid += 1

    # IMAGE CHUNKS
    for item in image_items:
        try:
            ocr_text = ocr.run(item["path"])
        except:
            ocr_text = ""

        chunks.append({
            "id": cid,
            "type": "image",
            "content": ocr_text,
            "image_path": item["path"],
            "metadata": item
        })
        cid += 1

    return chunks


# ----------------------------
# VECTOR DB
# ----------------------------
class VectorDB:
    def __init__(self, dim=768, path=None):
        if path is None:
            path = os.path.join(BASE, "db")
        os.makedirs(path, exist_ok=True)

        self.dim = dim
        self.index_path = os.path.join(path, "faiss.index")
        self.meta_path = os.path.join(path, "meta.pkl")

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.meta = pickle.load(open(self.meta_path, "rb"))
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

    def add(self, vec, meta):
        self.index.add(np.array([vec]).astype("float32"))
        self.meta.append(meta)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        pickle.dump(self.meta, open(self.meta_path, "wb"))

    def search(self, qvec, k=5):
        D, I = self.index.search(np.array([qvec]).astype("float32"), k)
        results = []
        for i in range(k):
            if I[0][i] == -1:
                continue
            results.append({
                "score": float(D[0][i]),
                "metadata": self.meta[I[0][i]]
            })
        return results


# ----------------------------
# BUILD INDEX
# ----------------------------
def build_index(pdf_folder=None):
    if pdf_folder is None:
        pdf_folder = os.path.join(BASE, "data")

    text_emb = TextEmbedder()
    img_emb = ImageEmbedder()
    db = VectorDB(768)

    all_chunks = []

    for pdf in os.listdir(pdf_folder):
        if pdf.endswith(".pdf"):
            t_items, i_items = extract_pdf(os.path.join(pdf_folder, pdf))
            all_chunks.extend(prepare_chunks(t_items, i_items))

    for ch in all_chunks:
        if ch["type"] == "image":
            vec = img_emb.embed(ch["image_path"])
        else:
            vec = text_emb.embed(ch["content"])

        db.add(vec, ch)

    db.save()
    return db


# ----------------------------
# RAG QUERY
# ----------------------------
def rag_query(query_text=None, image_path=None, k=5):
    text_emb = TextEmbedder()
    img_emb = ImageEmbedder()
    db = VectorDB()
    llm = LLM()

    # query embed
    if image_path:
        qvec = img_emb.embed(image_path)
    else:
        qvec = text_emb.embed(query_text)

    start = time.time()
    results = db.search(qvec, k)
    latency = time.time() - start

    # build context
    context = ""
    for r in results:
        meta = r["metadata"]["metadata"]
        context += f"[{meta['source']} Page {meta['page']}]\n{r['metadata']['content']}\n\n"

    # LLM prompt
    prompt = f"""
You are an AI assistant answering based on provided PDF context.

Context:
{context}

Question:
{query_text}

Answer clearly and concisely:
"""

    answer = llm.generate(prompt)
    return results, answer, round(latency, 3)
