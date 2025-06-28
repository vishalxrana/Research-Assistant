"""
Services for interacting with ChromaDB and the Gemini API.
"""
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from app.config import (
    EMBEDDING_MODEL_NAME,
    GENERATIVE_MODEL_NAME,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    GEMINI_API_KEY,
)

# Initialize models and clients
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME,
    device="cpu"
)

chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=chroma_embedding_function
)

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your environment or a .env file.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GENERATIVE_MODEL_NAME)


def upsert_chunks(chunks: List[Dict[str, Any]]):
    """Upserts a list of chunks into the ChromaDB collection."""
    ids = [chunk['id'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    chroma_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


def query_chroma(query: str, k: int) -> Dict[str, Any]:
    """Queries the ChromaDB collection for similar chunks."""
    return chroma_collection.query(
        query_texts=[query],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )


def get_all_chunks_from_chroma() -> Dict[str, Any]:
    """Retrieves all chunks from the ChromaDB collection."""
    return chroma_collection.get(include=['metadatas'])


def get_chunks_by_journal_id(journal_id: str) -> Dict[str, Any]:
    """Retrieves all chunks for a specific journal from the ChromaDB collection."""
    return chroma_collection.get(
        where={"source_doc_id": journal_id},
        include=['documents', 'metadatas']
    )


def update_usage_counts_in_chroma(chunk_ids: List[str]):
    """Updates the usage_count for a list of chunk IDs in the ChromaDB collection."""
    for chunk_id in chunk_ids:
        try:
            current_chunk_data = chroma_collection.get(ids=[chunk_id], include=['metadatas'])
            if current_chunk_data and current_chunk_data['metadatas']:
                current_metadata = current_chunk_data['metadatas'][0]
                current_usage_count = current_metadata.get("usage_count", 0)
                new_usage_count = current_usage_count + 1
                chroma_collection.update(
                    ids=[chunk_id],
                    metadatas=[{"usage_count": new_usage_count}]
                )
        except Exception as e:
            print(f"Error updating usage_count for chunk {chunk_id}: {e}")


def generate_llm_response(prompt: str) -> str:
    """Generates a response from the generative AI model."""
    response = gemini_model.generate_content(prompt)
    return response.text
