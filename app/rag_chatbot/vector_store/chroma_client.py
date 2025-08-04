"""
chroma_client.py

Stellt die Verbindung zur externen ChromaDB über REST her
und liefert ein LangChain-kompatibles Vectorstore-Objekt.
"""

import logging
from langchain_community.vectorstores import Chroma
from ..config import Config

logger = logging.getLogger(__name__)


def get_chroma_vectorstore(embedding_model) -> Chroma:
    """
    Erstellt eine Verbindung zu ChromaDB via LangChain.
    Übergibt ein Embedding-Modell von außen.
    """
    try:
        vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            client_settings={
                "chroma_api_impl": "rest",
                "chroma_server_host": Config.CHROMA_HOST,
                "chroma_server_http_port": Config.CHROMA_PORT,
            },
        )
        logger.info("✅ ChromaDB verbunden unter %s", Config.CHROMA_HTTP_URL)
        return vectorstore

    except Exception as e:
        logger.error("❌ Fehler beim Verbinden mit ChromaDB: %s", e)
        raise
