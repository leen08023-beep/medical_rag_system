from pathlib import Path
import re

def clean_text(text):
    """
    Remove weird Unicode characters and normalize spaces.
    """
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=700, overlap=100):
    """
    Splits text into overlapping chunks to preserve context.
    Medical terms are less likely to be lost at the edges this way.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def load_documents():
    """
    Load medical text, clean it, and split into smart chunks.
    """
    file_path = Path("data/pmc_diabetes.txt")
    
    if not file_path.exists():
        print(f"❌ Error: {file_path} not found.")
        return []

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    
    # Using sliding window instead of splitting on ". "
    paragraphs = chunk_text(text)

    return paragraphs

if __name__ == "__main__":
    docs = load_documents()
    print(f"✅ Created {len(docs)} chunks.")
    if docs:
        print("--- Sample Chunk ---")
        print(docs[0][:200] + "...")