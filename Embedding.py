import os
import json
import sqlite3
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

# === CONFIG ===
DISCOURSE_JSON_PATH = "discourse_posts.json"
MARKDOWN_DIR = "markdown_files"
DB_PATH = "knowledge_base.db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === EMBEDDING MODEL ===
model = SentenceTransformer(EMBEDDING_MODEL)

# === SETUP ===
def create_connection():
    return sqlite3.connect(DB_PATH)

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT,
            url TEXT,
            content TEXT,
            embedding BLOB
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()

# === TEXT CLEANING AND CHUNKING ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)

# === DISCOURSE PROCESSING ===
def process_discourse(conn):
    if not os.path.exists(DISCOURSE_JSON_PATH):
        print(f"❌ {DISCOURSE_JSON_PATH} not found.")
        return
    
    with open(DISCOURSE_JSON_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)

    chunks = []
    metadata = []

    for post in posts:
        content = clean_text(post.get("content", ""))
        if len(content) < 30:
            continue
        post_chunks = chunk_text(content)
        chunks.extend(post_chunks)
        metadata.extend([{
            "post_id": str(post.get("post_id")),
            "url": post.get("url")
        }] * len(post_chunks))

    embeddings = embed_chunks(chunks)

    cursor = conn.cursor()
    for i in range(len(chunks)):
        emb_blob = json.dumps(embeddings[i].tolist()).encode()
        cursor.execute("""
            INSERT INTO discourse_chunks (post_id, url, content, embedding)
            VALUES (?, ?, ?, ?)
        """, (metadata[i]['post_id'], metadata[i]['url'], chunks[i], emb_blob))
    conn.commit()
    print(f"✅ Processed {len(chunks)} chunks from Discourse")

# === MARKDOWN PROCESSING ===
def process_markdown(conn):
    if not os.path.exists(MARKDOWN_DIR):
        print(f"❌ {MARKDOWN_DIR} directory not found.")
        return

    files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith(".md")]
    if not files:
        print("❌ No markdown files found.")
        return

    chunks = []
    metadata = []

    for file in files:
        path = os.path.join(MARKDOWN_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            content = clean_text(f.read())
            if len(content) < 30:
                continue
            file_chunks = chunk_text(content)
            chunks.extend(file_chunks)
            metadata.extend([{
                "file_name": file,
                "chunk_index": i
            } for i in range(len(file_chunks))])

    embeddings = embed_chunks(chunks)

    cursor = conn.cursor()
    for i in range(len(chunks)):
        emb_blob = json.dumps(embeddings[i].tolist()).encode()
        cursor.execute("""
            INSERT INTO markdown_chunks (file_name, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?)
        """, (metadata[i]['file_name'], metadata[i]['chunk_index'], chunks[i], emb_blob))
    conn.commit()
    print(f"✅ Processed {len(chunks)} chunks from markdown files")

# === MAIN ===
def main():
    conn = create_connection()
    create_tables(conn)
    process_discourse(conn)
    process_markdown(conn)
    conn.close()
    print("🎉 All preprocessing complete.")

if __name__ == "__main__":
    main()
