# utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import io
import base64

# ---------------------------------------------------------
# SIMPLE TEXT CHUNKER
# ---------------------------------------------------------
def chunk_text(text, chunk_size=250):
    """
    Splits long text into chunks of `chunk_size` words.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# ---------------------------------------------------------
# RETRIEVAL METRICS
# ---------------------------------------------------------
def precision_at_k(retrieved, truth_ids, k):
    retrieved_ids = [r["metadata"].get("id") for r in retrieved[:k]]
    retrieved_ids = [x for x in retrieved_ids if x is not None]
    if k == 0:
        return 0
    return len(set(retrieved_ids) & set(truth_ids)) / k


def recall_at_k(retrieved, truth_ids, k):
    if len(truth_ids) == 0:
        return 0
    retrieved_ids = [r["metadata"].get("id") for r in retrieved[:k]]
    return len(set(retrieved_ids) & set(truth_ids)) / len(truth_ids)


def mean_avg_precision(retrieved, truth_ids):
    score = 0
    relevance = 0

    for i, r in enumerate(retrieved):
        if r["metadata"].get("id") in truth_ids:
            relevance += 1
            score += relevance / (i + 1)

    if len(truth_ids) == 0:
        return 0

    return score / len(truth_ids)


# ---------------------------------------------------------
# 2D EMBEDDING VISUALIZATION FOR STREAMLIT
# ---------------------------------------------------------
def visualize_embeddings_2d(vectors, labels=None):
    """
    Runs TSNE on vectors and returns a base64 PNG for Streamlit.
    """

    if len(vectors) == 0:
        return None

    # Reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=200)
    points = tsne.fit_transform(np.array(vectors))

    # Create plot
    plt.figure(figsize=(6, 5))
    plt.scatter(points[:, 0], points[:, 1], s=10, c='blue')

    if labels is not None:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (points[i, 0], points[i, 1]), fontsize=6)

    plt.title("2D Embedding Space")
    plt.tight_layout()

    # Convert plot → PNG → base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode()
    plt.close()

    # Return embeddable HTML image
    return f"data:image/png;base64,{encoded}"


# utils.py (add at bottom)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import base64
from io import BytesIO
import numpy as np
import faiss
import os


def visualize_embeddings_2d(vecs):
    """
    vecs: numpy array of shape (N, dim)
    returns Base64 image for Streamlit.
    """

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    # Step 1: PCA -> 50 dims
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(vecs)

    # Step 2: t-SNE 2D (use max_iter instead of n_iter)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,       # <-- FIXED HERE
        init="random"
    )
    emb_2d = tsne.fit_transform(reduced)

    # Step 3: Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10, alpha=0.7)
    ax.set_title("Embedding Space (t-SNE)")
    ax.set_xticks([])
    ax.set_yticks([])

    # Step 4: Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return img_base64
