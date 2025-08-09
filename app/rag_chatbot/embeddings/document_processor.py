import logging
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from ..config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.data_path = Config.DATA_DIR

        self.textsplitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def process_files(self, filename: str) -> List[Document]:
        """
        Lädt eine Datei und chunked sie
        """
        file_path = self.data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        try:
            # Laden des Textinhalts aus der Datei
            # TextLoader wandelt die Textdatei in ein Dokument-Objekt um, das von LangChain verarbeitet werden kann
            loader = TextLoader(file_path)
            documents = loader.load() # Lädt den gesamten Text als Dokument

            # Verarbeitung der Texte als Chunks
            docs = self.textsplitter.split_documents(documents)

            logger.info("Datei %s: %d Chunks erstellt", filename, len(docs))
            return docs
        
        except Exception as e:
            logger.error("Fehler beim Verarbeiten von %s: %s", filename, e)
            raise