from fastapi import FastAPI, APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app import services, config

app = FastAPI()
router = APIRouter()

class Chunk(BaseModel):
    id: str
    source_doc_id: str
    chunk_index: int
    section_heading: str
    doi: str | None = None
    journal: str
    publish_year: int
    usage_count: int
    attributes: List[str]
    link: str
    text: str

class SimilaritySearchRequest(BaseModel):
    query: str
    k: int = 10
    min_score: float = 0.25


@router.put("/upload")
async def upload_chunks(chunks: List[Chunk]):
    """Uploads a list of journal chunks to the ChromaDB collection."""
    try:
        chunk_data = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": {
                    "source_doc_id": chunk.source_doc_id,
                    "chunk_index": chunk.chunk_index,
                    "section_heading": chunk.section_heading,
                    "journal": chunk.journal,
                    "publish_year": chunk.publish_year,
                    "usage_count": chunk.usage_count,
                    "attributes": ",".join(chunk.attributes),
                    "link": chunk.link,
                    "doi": chunk.doi or "",
                    "id": chunk.id,
                },
            }
            for chunk in chunks
        ]
        services.upsert_chunks(chunk_data)
        return {"message": f"Successfully uploaded and processed {len(chunks)} chunks to ChromaDB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add chunks to ChromaDB: {e}")


@router.post("/similarity_search")
async def similarity_search(request_body: SimilaritySearchRequest):
    """Performs a similarity search on the ChromaDB collection."""
    try:
        results = services.query_chroma(request_body.query, request_body.k)
        raw_hits = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                score = 1 - distance
                if score >= request_body.min_score:
                    raw_hits.append({
                        "chunk_id": metadata.get("id"),
                        "source_doc_id": metadata.get("source_doc_id"),
                        "text": doc,
                        "score": score,
                    })
        formatted_results = sorted(raw_hits, key=lambda x: x["score"], reverse=True)[:request_body.k]
        chunk_ids_to_update = [hit["chunk_id"] for hit in formatted_results if hit.get("chunk_id")]
        if chunk_ids_to_update:
            services.update_usage_counts_in_chroma(chunk_ids_to_update)
        return {
            "query": request_body.query,
            "k": request_body.k,
            "min_score": request_body.min_score,
            "results": formatted_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {e}")


app.include_router(router, prefix=config.API_PREFIX)


@app.get("/")
def read_root():
    return {"Hello": "World"}