import chromadb
from chromadb.config import Settings

def get_collection(collection_name: str = "finrag"):
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks(collection, chunks: list[dict]):
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )
    return len(chunks)

def query_collection(collection, embedding: list[float], top_k: int = 20):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 4)
        }
        for i in range(len(results["documents"][0]))
    ]

def get_collection_stats(collection):
    return {"document_count": collection.count()}