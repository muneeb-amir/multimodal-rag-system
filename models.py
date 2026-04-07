# models.py

import os
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
from dotenv import load_dotenv
import openai

# Load .env
load_dotenv()

# OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------
# OCR MODEL
# -------------------------
class OCRModel:
    def run(self, path):
        try:
            return pytesseract.image_to_string(Image.open(path))
        except:
            return ""


# -------------------------
# Projection Layer (to unify dims)
# -------------------------
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.net(x).detach().cpu().numpy()


# -------------------------
# TEXT EMBEDDER (SBERT → 768)
# -------------------------
class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dim
        self.projector = ProjectionLayer(384)

    def embed(self, text):
        vec = self.model.encode(text, convert_to_numpy=True)
        return self.projector(vec)


# -------------------------
# IMAGE EMBEDDER (CLIP → 768)
# -------------------------
class ImageEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.projector = ProjectionLayer(512)

    def embed(self, path):
        img = Image.open(path).convert("RGB")
        inp = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self.model.get_image_features(**inp)
        vec = out.squeeze().cpu().numpy()
        return self.projector(vec)


# -------------------------
# LLM USING OPENAI GPT-4o MINI
# -------------------------

class LLM:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt, model="gpt-4o-mini"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]
