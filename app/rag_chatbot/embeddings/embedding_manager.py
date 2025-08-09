import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config import Config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self):
        self.config = Config()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lädt das Embedding-Modell mit LangChain"""
        try:
            self.model = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
            logger.info(
                "LangChain-Embedding-Modell geladen: %s", self.config.EMBEDDING_MODEL
            )
        except Exception as e:
            logger.error("Fehler beim Laden des LangChain-Modells: %s", e)
            raise

    def get_model(self):
        """Gibt das LangChain-kompatible Embedding-Modell zurück"""
        return self.model
