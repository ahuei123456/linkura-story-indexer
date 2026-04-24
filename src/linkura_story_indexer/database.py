import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load API key from .env
load_dotenv()

def initialize_settings():
    """Initializes the LLM and Embedding models for LlamaIndex."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    
    # Configure Google GenAI LLM and Embedding model
    Settings.llm = GoogleGenAI(api_key=api_key, model="models/gemini-3-flash-preview")
    Settings.embed_model = GoogleGenAIEmbedding(api_key=api_key, model_name="models/gemini-embedding-2", embed_batch_size=1)


    Settings.chunk_size = 1024
    Settings.chunk_overlap = 50

def get_vector_store(collection_name: str = "story_nodes"):
    """Initializes ChromaDB and returns a VectorStoreIndex."""
    # Ensure local directory for ChromaDB storage
    db_path = "./chroma_db"
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return vector_store, storage_context
