import os
import time
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chroma_client import get_chroma_vectorstore
from config import Config

# Embedding-Instanz erstellen
emb = OllamaEmbeddings(
    base_url=Config.OLLAMA_BASE_URL,
    model=Config.OLLAMA_EMBEDDING_MODEL,
    keep_alive=300,
)

def read_files_from_directory(directory_path):
    """
    Liest alle Dateien aus einem Verzeichnis ein
    """
    documents = []
    if not os.path.exists(directory_path):
        print(f"❌ Verzeichnis {directory_path} existiert nicht!")
        return documents
   
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                loader = TextLoader(
                    file_path,
                    autodetect_encoding=True
                )
                file_documents = loader.load()
                documents.extend(file_documents)
                print(f"✅ Datei {filename} erfolgreich geladen.")
            except Exception as e:
                print(f"❌ Fehler beim Laden der Datei {filename}: {e}")
        else:
            print(f"⚠️ {filename} ist keine Datei und wird übersprungen.")
    return documents

def embed_documents_in_batches(documents, batch_size=50, max_retries=3):
    """
    Splittet Dokumente in Chunks und embedded sie in kleineren Batches in die Chroma DB
    """
    if not documents:
        print("❌ Keine Dokumente zum Embedden gefunden!")
        return
   
    print(f"📄 Verarbeite {len(documents)} Dokumente...")
   
    # 1. Text Splitting mit Config-Parametern
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
   
    print("✂️ Teile Dokumente in Chunks auf...")
    chunks = text_splitter.split_documents(documents)
    print(f"📝 {len(chunks)} Text-Chunks erstellt")
   
    # Optional: Zeige ersten Chunk als Beispiel
    if chunks:
        print(f"📋 Beispiel-Chunk:\n{chunks[0].page_content[:200]}...\n")
   
    # 2. ChromaDB initialisieren
    try:
        print("🔧 Initialisiere ChromaDB...")
        db = get_chroma_vectorstore(emb)
        print("✅ ChromaDB-Verbindung hergestellt")
    except Exception as e:
        print(f"❌ Fehler bei ChromaDB-Initialisierung: {e}")
        return
   
    # 3. Chunks in Batches verarbeiten
    total_chunks = len(chunks)
    num_batches = (total_chunks + batch_size - 1) // batch_size
    
    print(f"🔄 Verarbeite {total_chunks} Chunks in {num_batches} Batches (Batch-Größe: {batch_size})")
    
    successful_batches = 0
    failed_batches = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        
        print(f"📦 Batch {batch_idx + 1}/{num_batches}: Chunks {start_idx + 1}-{end_idx} ({len(batch_chunks)} Chunks)")
        
        # Retry-Mechanismus für jeden Batch
        retry_count = 0
        batch_success = False
        
        while retry_count < max_retries and not batch_success:
            try:
                # Embeddings erstellen und speichern
                db.add_documents(batch_chunks)
                print(f"✅ Batch {batch_idx + 1} erfolgreich verarbeitet")
                successful_batches += 1
                batch_success = True
                
                # Kurze Pause zwischen Batches um Server zu entlasten
                if batch_idx < num_batches - 1:  # Nicht nach dem letzten Batch warten
                    time.sleep(2)
                    
            except Exception as e:
                retry_count += 1
                print(f"❌ Fehler in Batch {batch_idx + 1}, Versuch {retry_count}/{max_retries}: {e}")
                
                if retry_count < max_retries:
                    wait_time = 5 * retry_count  # Exponential backoff
                    print(f"⏳ Warte {wait_time} Sekunden vor erneutem Versuch...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Batch {batch_idx + 1} endgültig fehlgeschlagen nach {max_retries} Versuchen")
                    failed_batches += 1
    
    # Zusammenfassung
    print(f"\n📊 Embedding-Zusammenfassung:")
    print(f"✅ Erfolgreiche Batches: {successful_batches}/{num_batches}")
    print(f"❌ Fehlgeschlagene Batches: {failed_batches}/{num_batches}")
    
    if successful_batches > 0:
        try:
            collection = db._collection
            total_docs_in_db = collection.count()
            print(f"📚 Gesamtanzahl Dokumente in Collection '{Config.CHROMA_COLLECTION_NAME}': {total_docs_in_db}")
        except:
            print("📚 Collection-Info nicht verfügbar")
    
    if failed_batches == 0:
        print("🎉 Alle Chunks erfolgreich embedded!")
    elif failed_batches < num_batches:
        print("⚠️ Embedding teilweise erfolgreich - einige Batches sind fehlgeschlagen")
    else:
        print("❌ Embedding komplett fehlgeschlagen")

def main():
    """
    Hauptfunktion
    """
    print("🚀 Starte Embedding-Prozess...")
   
    # 1. Dateien laden
    data_path = "/app/data/raw_data_books"
    documents = read_files_from_directory(data_path)
   
    if not documents:
        print("❌ Keine Dokumente gefunden!")
        return
   
    # 2. Embedden mit Batch-Processing
    # Batch-Größe anpassen je nach Server-Performance
    # Kleinere Batches = stabiler aber langsamer
    batch_size = 50  # Reduzieren auf 25 oder 10 falls weiterhin Timeouts
    embed_documents_in_batches(documents, batch_size=batch_size)
   
    print("🎉 Embedding-Prozess abgeschlossen!")

if __name__ == "__main__":
    main()