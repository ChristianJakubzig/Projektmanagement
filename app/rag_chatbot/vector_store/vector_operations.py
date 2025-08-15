import logging
import gc
import time
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class vector_operations:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 8,  # NOCH kleinere Batches!
        persist_every: int = 1,  # Nach jedem Batch
        sleep_between_batches: float = 0.5  # Kurze Pause zwischen Batches
    ) -> None:
        """
        Fügt Dokumente in sehr kleinen Batches zur ChromaDB hinzu.
        Ultra-aggressive Speicheroptimierung.
        """
        n = len(documents)
        logger.info(f"➡️ Starte Embedding von {n} Dokumenten in Mini-Batches à {batch_size}")
        batches_done = 0
       
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            
            # Batch erstellen - NUR die Referenzen, nicht kopieren
            batch = documents[start:end]
           
            try:
                # Embedding durchführen
                self.vectorstore.add_documents(batch)
                batches_done += 1
                batch_len = len(batch)
               
                # SOFORTIGE aggressive Speicherbereinigung
                batch.clear()  # Liste leeren (falls möglich)
                del batch
                
                # Mehrfache Garbage Collection
                gc.collect()
                gc.collect()  # Ja, zweimal!
               
                logger.info(f"✅ Mini-Batch {batches_done}: {batch_len} Docs ({end}/{n})")
                print(f"✅ Mini-Batch {batches_done}: {batch_len} Chunks embeddet ({end}/{n})")
               
                # Nach JEDEM Batch persistieren
                if hasattr(self.vectorstore, "persist"):
                    self.vectorstore.persist()
                    logger.info("💾 Persist nach Mini-Batch.")
                
                # Kurze Pause - gibt dem System Zeit für Cleanup
                if sleep_between_batches > 0:
                    time.sleep(sleep_between_batches)
                   
            except Exception as e:
                logger.error(f"❌ Fehler in Mini-Batch {batches_done} ({start}:{end}): {e}")
                # Cleanup auch bei Fehler
                try:
                    batch.clear()
                    del batch
                    gc.collect()
                    gc.collect()
                except:
                    pass
                raise
       
        # Finaler Persist (zur Sicherheit)
        if hasattr(self.vectorstore, "persist"):
            self.vectorstore.persist()
            logger.info("💾 Finales Persist abgeschlossen.")
           
        print(f"🎉 Ultra-speicherschonend: {n} Chunks erfolgreich embeddet!")