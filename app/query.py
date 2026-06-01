import os
import cohere
import anthropic
import time
from dotenv import load_dotenv
from app.embed import embed_query
from app.store import get_collection, query_collection

load_dotenv()

cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a financial document analyst. Answer questions based strictly on the provided context from financial documents.

Rules:
- Only use information from the provided context
- Always cite which document your answer comes from
- If the context does not contain enough information, say so explicitly
- Be precise and concise
- Never hallucinate or make up information"""

def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    
    response = cohere_client.rerank(
        query=query,
        documents=[c["text"] for c in chunks],
        top_n=top_n,
        model="rerank-english-v3.0"
    )
    
    reranked = []
    for hit in response.results:
        chunk = chunks[hit.index]
        chunk["rerank_score"] = round(hit.relevance_score, 4)
        reranked.append(chunk)
    
    return reranked

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(
            f"[Source {i+1}: {source}]\n{chunk['text']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    return f"""Context from financial documents:

{context}

---

Question: {question}

Answer based strictly on the context above. Cite sources by their [Source N] label."""

def query_documents(question: str, top_k: int = 5) -> dict:
    start = time.time()
    
    # Step 1: embed the question
    question_embedding = embed_query(question)
    
    # Step 2: retrieve top 20 from vector store
    collection = get_collection()
    raw_chunks = query_collection(collection, question_embedding, top_k=20)
    
    if not raw_chunks:
        return {
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "latency_ms": 0
        }
    
    # Step 3: rerank to top_k
    reranked_chunks = rerank(question, raw_chunks, top_n=top_k)
    
    # Step 4: build prompt
    prompt = build_prompt(question, reranked_chunks)
    
    # Step 5: call Claude
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = message.content[0].text
    latency = round((time.time() - start) * 1000)
    
    # Step 6: format sources
    sources = [
        {
            "source": c["metadata"].get("source", "unknown"),
            "chunk_index": c["metadata"].get("chunk_index"),
            "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
            "retrieval_score": c.get("score"),
            "rerank_score": c.get("rerank_score")
        }
        for c in reranked_chunks
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency
    }