import logging
import gc
from typing import List
from langchain.text_splitter import CharacterTextSplitter  # Original beibehalten
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from ..config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.data_path = Config.DATA_DIR
        # Original CharacterTextSplitter beibehalten
        self.textsplitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
    
    def process_files(self, filename: str) -> List[Document]:
        """Lädt eine Datei und chunked sie"""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
       
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            docs = self.textsplitter.split_documents(documents)
            logger.info("Datei %s: %d Chunks erstellt", filename, len(docs))
            return docs
        except Exception as e:
            logger.error("Fehler beim Verarbeiten von %s: %s", filename, e)
            raise
    
    def process_and_embed_files(self, vector_ops, max_chunks_per_run: int = 50) -> None:
        """
        Ultra-speicherschonende Verarbeitung:
        - Verarbeitet nur max_chunks_per_run Chunks auf einmal
        - Bei großen Dateien: Stoppt nach X Chunks und startet neu
        """
        txt_files = list(self.data_path.glob("*.txt"))
        if not txt_files:
            logger.warning("Keine .txt Dateien in %s gefunden", self.data_path)
            return
        
        logger.info("Ultra-speicherschonende Verarbeitung von %d Dateien...", len(txt_files))
        total_chunks = 0
        
        for file_path in txt_files:
            try:
                print(f"📖 Verarbeite {file_path.name}...")
                docs = self.process_files(file_path.name)
                file_chunks = len(docs)
                total_chunks += file_chunks
                print(f"✅ {file_path.name}: {file_chunks} Chunks")
                
                # Große Dateien in kleinere Portionen aufteilen
                if file_chunks > max_chunks_per_run:
                    print(f"⚠️  Große Datei! Teile in {max_chunks_per_run}er Portionen auf...")
                    
                    for start in range(0, file_chunks, max_chunks_per_run):
                        end = min(start + max_chunks_per_run, file_chunks)
                        chunk_portion = docs[start:end]
                        portion_size = len(chunk_portion)
                        
                        print(f"  📦 Portion {start//max_chunks_per_run + 1}: {portion_size} Chunks...")
                        vector_ops.add_documents(chunk_portion)
                        
                        # Aggressive Bereinigung nach jeder Portion
                        del chunk_portion
                        gc.collect()
                        gc.collect()
                        
                        print(f"  ✅ Portion verarbeitet und Speicher bereinigt")
                else:
                    # Kleine Datei - normal verarbeiten
                    vector_ops.add_documents(docs)
                
                # Datei komplett verarbeitet - aggressives Cleanup
                del docs
                gc.collect()
                gc.collect()
                print(f"🧹 Datei-Cleanup für {file_path.name} abgeschlossen")
               
            except Exception as e:
                print(f"❌ Fehler bei {file_path.name}: {e}")
                # Cleanup auch bei Fehler
                try:
                    del docs
                    gc.collect()
                except:
                    pass
                continue
        
        print(f"\n🎉 Ultra-speicherschonend abgeschlossen: {total_chunks} Chunks aus {len(txt_files)} Dateien")
    
    def process_all_files(self) -> List[Document]:
        """Original-Methode - NICHT für große Datenmengen verwenden!"""
        print("⚠️  WARNUNG: process_all_files() kann bei großen Datenmengen OOM verursachen!")
        
        all_docs = []
        txt_files = list(self.data_path.glob("*.txt"))
        if not txt_files:
            logger.warning("Keine .txt Dateien in %s gefunden", self.data_path)
            return []
       
        logger.info("Verarbeitet %d Dateien...", len(txt_files))
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
