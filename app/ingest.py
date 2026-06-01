import os
import uuid
import pypdf
from dotenv import load_dotenv
from app.embed import embed_text
from app.store import get_collection, add_chunks

load_dotenv()

def load_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_path: str) -> dict:
    print(f"Loading {file_path}...")
    text = load_pdf(file_path)
    if not text.strip():
        return {"error": f"No text extracted from {file_path}"}

    print(f"Chunking...")
    chunks = chunk_text(text)
    print(f"Got {len(chunks)} chunks. Embedding...")

    collection = get_collection()
    file_name = os.path.basename(file_path)

    prepared = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        prepared.append({
            "id": str(uuid.uuid4()),
            "text": chunk,
            "embedding": embedding,
            "metadata": {
                "source": file_name,
                "chunk_index": i
            }
        })
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(chunks)} chunks...")

    add_chunks(collection, prepared)
    print(f"Done. {len(prepared)} chunks stored.")

    return {
        "file": file_name,
        "chunks_ingested": len(prepared),
        "status": "success"
    }

def ingest_all_pdfs(folder_path: str = "./data/docs") -> dict:
    pdf_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".pdf")
    ]
    if not pdf_files:
        return {"error": "No PDF files found in data/docs/"}

    results = []
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        result = ingest_pdf(full_path)
        results.append(result)

    total_chunks = sum(r.get("chunks_ingested", 0) for r in results)
    return {
        "files_processed": len(results),
        "total_chunks": total_chunks,
        "results": results
    }