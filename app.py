import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sqlite3
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

# === Load Environment ===
load_dotenv()

# === Config ===g
DB_PATH = "knowledge_base.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GGUF_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"  # adjust as needed
CTX_LEN = 4096  # context length

# === Load Models ===
embedder = SentenceTransformer(EMBEDDING_MODEL)

llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_ctx=CTX_LEN,
    n_threads=os.cpu_count(),  # use all CPU cores
    n_gpu_layers=0  # for CPU only; increase if you want GPU acceleration
)

# === FastAPI App ===
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Reserved for future use

# === Helpers ===

def get_embedding(text):
    return embedder.encode([text])[0]

def load_chunks(table_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, content, embedding FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()

    chunks = []
    for row in rows:
        id_, content, emb_blob = row
        emb = np.array(json.loads(emb_blob.decode()))
        chunks.append((id_, content, emb))
    return chunks

def find_best_match(question, table_name, top_k=2):
    q_emb = get_embedding(question)
    chunks = load_chunks(table_name)
    scored = [(id_, content, cosine_similarity([q_emb], [emb])[0][0]) for id_, content, emb in chunks]
    return sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]

def generate_answer(question, context):
    prompt = f"[INST] Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer: [/INST]"
    output = llm(prompt, max_tokens=512, stop=["</s>", "[/INST]"])
    return output["choices"][0]["text"].strip()

# === API Endpoint ===
@app.post("/api/")
async def answer_question(req: QuestionRequest):
    top_discourse = find_best_match(req.question, "discourse_chunks")
    top_markdown = find_best_match(req.question, "markdown_chunks")
    context = "\n".join([c for _, c, _ in top_discourse + top_markdown])

    answer = generate_answer(req.question, context)
    links = []  # optional: link to sources

    return {
        "answer": answer,
        "links": links
    }
