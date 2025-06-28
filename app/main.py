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

class ChatRequest(BaseModel):
    query: str
    k: int = 5
    min_score: float = 0.5


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


@router.post("/chat")
async def chat_with_llm(request_body: ChatRequest):
    """Engages in a chat with the generative AI."""
    try:
        results = services.query_chroma(request_body.query, request_body.k)
        
        context_chunks_with_source = []
        citations = set()
        chunk_ids_to_update = []

        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                source_id = metadata.get("source_doc_id", "Unknown")
                
                # Add the source to the context for the LLM
                context_chunks_with_source.append(f"Source ID: {source_id}\nContent: {doc}")
                
                citations.add(source_id)
                if metadata.get("id"):
                    chunk_ids_to_update.append(metadata.get("id"))

        if chunk_ids_to_update:
            services.update_usage_counts_in_chroma(chunk_ids_to_update)

        if not context_chunks_with_source:
            llm_prompt = f"Answer the following question: {request_body.query}"
            citations_list = []
        else:
            context_text = "\n\n---\n\n".join(context_chunks_with_source)
            llm_prompt = f"""You are a factual research assistant. Your task is to answer the user's question based ONLY on the provided context below.

The context consists of several chunks of text, each preceded by its 'Source ID'.

Instructions:
1. Carefully read the user's question and all the provided context.
2. Formulate a comprehensive answer to the question using ONLY the information from the context.
3. For EVERY piece of information you use from a source, you MUST cite it immediately after the statement using the format `[Source: <Source ID>]`.
4. If the answer requires information from multiple sources, cite each one.
5. If the provided context does not contain the answer, you MUST state: "I could not find an answer in the provided documents." Do not use any outside knowledge.

Context:
{context_text}

Question: {request_body.query}

Answer:
"""
            citations_list = list(citations)

        full_response = services.generate_llm_response(llm_prompt)
        
        # The main response should already contain inline citations. 
        # The 'citations' list at the end is a summary of unique sources used.
        return {"answer": full_response, "citations": citations_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat with LLM failed: {e}")


@router.get("/usage_statistics")
async def get_usage_statistics():
    """Retrieves usage statistics for all documents."""
    try:
        all_chunks = services.get_all_chunks_from_chroma()
        usage_data = {}
        if all_chunks and all_chunks['metadatas']:
            for metadata in all_chunks['metadatas']:
                source_doc_id = metadata.get("source_doc_id")
                usage_count = metadata.get("usage_count", 0)
                if source_doc_id:
                    usage_data[source_doc_id] = usage_data.get(source_doc_id, 0) + usage_count
        sorted_usage = sorted([
            {"source_doc_id": doc_id, "total_usage_count": count}
            for doc_id, count in usage_data.items()
        ], key=lambda x: x["total_usage_count"], reverse=True)
        return sorted_usage
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve usage statistics: {e}")


@router.get("/{journal_id}")
async def get_journal_content(journal_id: str):
    """Retrieves all chunks for a specific journal."""
    try:
        results = services.get_chunks_by_journal_id(journal_id)
        if not results or not results['documents']:
            raise HTTPException(status_code=404, detail=f"Journal with ID '{journal_id}' not found.")
        matching_chunks = []
        for i in range(len(results['documents'])):
            doc = results['documents'][i]
            metadata = results['metadatas'][i]
            attributes_list = metadata.get("attributes", "").split(",") if metadata.get("attributes") else []
            matching_chunks.append(Chunk(
                id=metadata.get("id"),
                source_doc_id=metadata.get("source_doc_id"),
                chunk_index=metadata.get("chunk_index"),
                section_heading=metadata.get("section_heading"),
                doi=metadata.get("doi"),
                journal=metadata.get("journal"),
                publish_year=metadata.get("publish_year"),
                usage_count=metadata.get("usage_count"),
                attributes=attributes_list,
                link=metadata.get("link"),
                text=doc,
            ))
        return matching_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve journal content: {e}")


app.include_router(router, prefix=config.API_PREFIX)


@app.get("/")
def read_root():
    return {"Hello": "World"}