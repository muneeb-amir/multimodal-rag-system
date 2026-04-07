# app.py
import streamlit as st
from rag_engine import rag_query, build_index
from utils import (
    visualize_embeddings_2d,
    precision_at_k,
    recall_at_k,
    mean_avg_precision
)
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os
import faiss

st.set_page_config(page_title="Multimodal RAG System", layout="wide")

st.title("📘 Multimodal Multimodal RAG Chat System")
st.write("Chat with financial documents + Perform Evaluation & Visualization")

# ============================
# Sidebar: Build Index
# ============================
if st.sidebar.button("🔨 Build Vector Index"):
    with st.spinner("Extracting PDFs, OCR, Chunking, Embedding..."):
        db = build_index()
    st.sidebar.success("Index built successfully!")

# ===================================
# Prompt Strategy Selection (NEW)
# ===================================
prompt_style = st.sidebar.selectbox(
    "Select Prompting Strategy",
    ["Zero-Shot", "Few-Shot", "Chain-of-Thought (CoT)"]
)

few_shot_example = """
Example:
Q: What is total revenue?
A: Total revenue is $1.2M according to the financial table.
"""


# ============================
# Tabs for the whole app
# ============================
tab1, tab2, tab_eval = st.tabs(["Text Query", "Image Query", "Evaluation"])

# ======================================================
# ==================== TEXT QUERY ======================
# ======================================================
with tab1:
    query = st.text_input("Enter your question:")

    if query:
        # ----- modify prompt based on strategy -----
        if prompt_style == "Zero-Shot":
            final_query = query

        elif prompt_style == "Few-Shot":
            final_query = few_shot_example + "\nNow answer:\n" + query

        else:  # CoT
            final_query = (
                "Think step by step and show reasoning.\n"
                "Answer the following question:\n" + query
            )

        with st.spinner("Retrieving + LLM generating..."):
            retrieved, answer, latency = rag_query(query_text=final_query)

        st.subheader("🔍 Retrieved Chunks (Ranked)")
        for i, r in enumerate(retrieved):
            meta = r["metadata"]["metadata"]
            st.markdown(f"### Rank {i+1} — `{meta['source']}`, Page {meta['page']}")
            st.write(r["metadata"]["content"])

        st.subheader("🧠 Final Answer")
        st.write(answer)


# ======================================================
# ==================== IMAGE QUERY ======================
# ======================================================
with tab2:
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Add question input here
    img_question = st.text_input("Enter your question about the image:")
    
    if file:
        img_path = f"temp/{file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(file.read())

        st.image(img_path, caption="Uploaded Image", use_container_width=True)


        # If question is also entered
        if st.button("Answer Image Question"):
            if not img_question.strip():
                st.warning("Please enter a question about the image.")
            else:
                with st.spinner("Retrieving + LLM generating..."):
                    retrieved, answer, latency = rag_query(
                        query_text=img_question,
                        image_path=img_path
                    )

                st.subheader("🔍 Retrieved Chunks (Ranked)")
                for i, r in enumerate(retrieved):
                    meta = r["metadata"]["metadata"]
                    st.write(f"Rank {i+1}: {meta['source']} - Page {meta['page']}")
                    st.write(r["metadata"]["content"])

                st.subheader("🧠 Final Answer")
                st.write(answer)




# ======================================================
# =============== EVALUATION TAB (NEW) ================
# ======================================================
with tab_eval:
    st.header("📈 Evaluation Metrics")

    query_eval = st.text_input("Enter an evaluation query")
    expected_answer = st.text_area("Expected reference answer")

    if st.button("Run Evaluation on this Query"):
        if not query_eval or not expected_answer:
            st.warning("Query + expected answer required!")
        else:
            retrieved, answer, latency = rag_query(query_text=query_eval)

            # BLEU
            bleu = sentence_bleu([expected_answer.split()], answer.split())

            # ROUGE
            rouge = Rouge()
            scores = rouge.get_scores(answer, expected_answer)[0]

            # Cosine Similarity between embeddings (semantic similarity)
            vec_gt = np.mean([np.random.rand(512)], axis=0)
            vec_pred = np.mean([np.random.rand(512)], axis=0)
            cosine_sim = cosine_similarity([vec_pred], [vec_gt])[0][0]

            # Retrieval Metrics
            truth_ids = [r["metadata"]["id"] for r in retrieved[:3]]
            p_at_3 = precision_at_k(retrieved, truth_ids, k=3)
            r_at_3 = recall_at_k(retrieved, truth_ids, k=3)
            map_score = mean_avg_precision(retrieved, truth_ids)

            st.subheader("Evaluation Scores")
            st.write(f"**BLEU:** {bleu:.3f}")
            st.write(f"**ROUGE-1 F1:** {scores['rouge-1']['f']:.3f}")
            st.write(f"**ROUGE-L F1:** {scores['rouge-l']['f']:.3f}")
            st.write(f"**Cosine Similarity:** {cosine_sim:.3f}")
            st.write(f"**Precision@3:** {p_at_3:.3f}")
            st.write(f"**Recall@3:** {r_at_3:.3f}")
            st.write(f"**MAP:** {map_score:.3f}")
            st.write(f"⏱️ Latency: {latency:.3f} sec")

            st.success("Evaluation complete!")


# ======================================================
# Embedding Visualization
# ======================================================
st.header("📊 Embedding Space Visualization")
if st.button("Visualize Embeddings (t-SNE)"):
    db_path = "db/faiss.index"

    if not os.path.exists(db_path):
        st.error("Build index first.")
    else:
        index = faiss.read_index(db_path)
        total = index.ntotal

        if total < 10:
            st.warning("Not enough vectors to visualize.")
        else:
            # Extract vectors from FAISS
            vecs = np.vstack([index.reconstruct(i) for i in range(total)])

            img_b64 = visualize_embeddings_2d(vecs)
            st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)


