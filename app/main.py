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


app.include_router(router, prefix=config.API_PREFIX)


@app.get("/")
def read_root():
    return {"Hello": "World"}