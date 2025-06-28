"""
Configuration settings for the application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Model Names
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATIVE_MODEL_NAME = 'gemini-2.0-flash' # Corrected model name

# ChromaDB Settings
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "journal_chunks"

# Application Settings
API_PREFIX = "/api"
