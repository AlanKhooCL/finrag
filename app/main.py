import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.ingest import ingest_all_pdfs
from app.query import query_documents
from app.store import get_collection, get_collection_stats

load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

app = FastAPI(
    title="FinRAG",
    description="Financial Document Intelligence API — RAG-powered Q&A over financial documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

async def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return key

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: int

class IngestResponse(BaseModel):
    files_processed: int
    total_chunks: int
    results: list[dict]

@app.get("/health")
async def health():
    collection = get_collection()
    stats = get_collection_stats(collection)
    return {
        "status": "ok",
        "model": "claude-sonnet-4-20250514",
        "embeddings": "google-text-embedding-004",
        "reranker": "cohere-rerank-english-v3.0",
        "version": "1.0.0",
        "vector_store": stats
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest(_=Depends(verify_api_key)):
    result = ingest_all_pdfs("./data/docs")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, _=Depends(verify_api_key)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = query_documents(req.question, req.top_k)
    return result