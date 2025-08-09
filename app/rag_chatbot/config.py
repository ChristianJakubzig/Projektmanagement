import os
from pathlib import Path
from typing import Optional

class Config:
    # ChromaDB
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))
    CHROMA_HTTP_URL: str = os.getenv("CHROMA_HTTP_URL") or f"http://{CHROMA_HOST}:{CHROMA_PORT}"
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "oliverguhr/revosax-granite-embedding-278m-multilingual")
    
    # LLM Config
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Collection Name
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "simple_embeddings")

    #Data Directory 
    DATA_DIR = Path(__file__).parent.parent /"data" / "raw_data_books"