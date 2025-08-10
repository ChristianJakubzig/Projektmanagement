import logging
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class vector_operations:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """
        Fügt Dokumente zur ChromaDB hinzu
        Das Embedding passiert automatisch

        Args:
            documents: Liste von Langchain Documents (mit text + metadaten)
        """
        try:
            self.vectorstore.add_documents(documents)
            
            logger.info(f"✅ {len(documents)} Dokumente zu ChromaDB hinzugefügt")
            print(f"✅ {len(documents)} Chunks erfolgreich embeddet und gespeichert!")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Hinzufügen der Dokumente: {e}")
            raise
        