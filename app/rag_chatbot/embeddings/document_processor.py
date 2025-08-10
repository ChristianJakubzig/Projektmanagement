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

    def process_all_files(self) -> List[Document]:
        """
        Verarbeitet alle Textdatein aus dem Ordner data/raw_data
        """
        all_docs = []

        txt_files = list(self.data_path.glob("*.txt"))

        if not txt_files:
            logger.warning("Keine .txt Dateien in %s gefunden", self.data_path)
            return []
        
        logger.info("Verarbeitet %d Datein...", len(txt_files))

        for file_path in txt_files:
            try:
                docs = self.process_files(file_path.name)
                all_docs.extend(docs)
                print(f"✅ {file_path.name}: {len(docs)} Chunks")

            except Exception as e:
                print(f"❌ Fehler bei {file_path.name}: {e}")
                continue

        print(f"\n🎉 Gesamt: {len(all_docs)} Chunks aus {len(txt_files)} Dateien")
        return all_docs
    
    def get_available_files(self) -> List[str]:
        """Liste aller verfügbaren Dateien"""
        txt_files = [f.name for f in self.data_path.glob("*.txt")]
        return sorted(txt_files)