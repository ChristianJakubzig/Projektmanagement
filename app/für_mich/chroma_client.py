# rag_chatbot/vector_store/chroma_client.py
import logging
from urllib.parse import urlparse
import chromadb
from langchain_chroma import Chroma  # NEU: statt langchain_community
from config import Config

logger = logging.getLogger(__name__)

def get_chroma_vectorstore(embedding_model) -> Chroma:
    """
    Verbindet sich mit Chroma v2 per REST (HttpClient) und liefert ein LangChain-Vectorstore.
    Nutzt Tenant/Database aus der Config (v2-Namespace).
    """
    try:
        u = urlparse(Config.CHROMA_HTTP_URL)
        ssl = (u.scheme == "https")
        port = u.port or (443 if ssl else 80)

        # v2: Tenant/Database an den HttpClient übergeben (wird von chromadb>=0.5/1.0 unterstützt)
        client = chromadb.HttpClient(
            host=u.hostname,
            port=port,
            ssl=ssl,
            tenant=getattr(Config, "CHROMA_TENANT", "default_tenant"),
            database=getattr(Config, "CHROMA_DATABASE", "default_database"),
        )

        vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model,
            client=client,  # wichtig: kein client_settings verwenden
        )

        logger.info("✅ ChromaDB v2 verbunden: %s (tenant=%s, db=%s)",
                    Config.CHROMA_HTTP_URL, getattr(Config, "CHROMA_TENANT", "default_tenant"),
                    getattr(Config, "CHROMA_DATABASE", "default_database"))
        return vectorstore

    except Exception as e:
        logger.error("❌ Fehler beim Verbinden mit ChromaDB: %s", e)
        raise
