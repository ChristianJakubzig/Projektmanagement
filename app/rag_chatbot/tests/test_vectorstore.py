"""
Test für die Vectorstore Funktionalität
"""
from rag_chatbot.embeddings.embedding_manager import EmbeddingManager
from rag_chatbot.vector_store.chroma_client import get_chroma_vectorstore

def test_show_collection_info():
    """Zeigt Collection-Informationen über den Vectorstore an"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
    
    # ChromaDB Collection-Objekt abrufen
    collection = vectorstore._collection
    
    print(f"\n📚 Collection-Informationen:")
    print(f"   Name: {collection.name}")
    print(f"   ID: {collection.id}")
    
    # Anzahl Dokumente
    try:
        count = collection.count()
        print(f"   📊 Anzahl Dokumente: {count}")
    except Exception as e:
        print(f"   📊 Anzahl Dokumente: Fehler - {e}")
    
    # Metadata der Collection
    try:
        metadata = collection.metadata
        if metadata:
            print(f"   🏷️ Metadata: {metadata}")
        else:
            print(f"   🏷️ Metadata: Keine")
    except Exception as e:
        print(f"   🏷️ Metadata: Fehler - {e}")
    
    # Ein paar Beispiel-Dokumente
    try:
        # Verwende peek() um ein paar Dokumente zu sehen ohne Embedding-Suche
        peek_result = collection.peek(limit=3)
        if peek_result and 'documents' in peek_result:
            print(f"\n   📄 Beispiel-Dokumente (erste 3):")
            for i, doc in enumerate(peek_result['documents'], 1):
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"      {i}. {preview}")
                
            # Auch IDs und Metadata zeigen falls vorhanden
            if 'ids' in peek_result and peek_result['ids']:
                print(f"\n   🆔 Beispiel-IDs:")
                for i, doc_id in enumerate(peek_result['ids'][:3], 1):
                    print(f"      {i}. {doc_id}")
                    
            if 'metadatas' in peek_result and peek_result['metadatas']:
                print(f"\n   🏷️ Beispiel-Metadaten:")
                for i, metadata in enumerate(peek_result['metadatas'][:3], 1):
                    if metadata:
                        print(f"      {i}. {metadata}")
        else:
            print(f"   📄 Keine Dokumente gefunden oder Collection leer")
    except Exception as e:
        print(f"   📄 Beispiel-Dokumente: Fehler - {e}")

def test_vectorstore_has_data():
    """Test dass die Datenbank Daten enthält"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
   
    # Erst Collection-Info anzeigen
    collection = vectorstore._collection
    count = collection.count()
    print(f"\n📊 Collection '{collection.name}' hat {count} Dokumente")
   
    # Test-Suche
    results = vectorstore.similarity_search("test", k=1)
   
    # Assertions
    assert len(results) > 0, "Datenbank ist leer - setup_database.py ausführen!"
    assert results[0].page_content, "Ergebnis hat keinen Inhalt"
    print(f"✅ Datenbank enthält {len(results)} Suchergebnisse")
    
    # Zeige Ergebnis-Details
    first_result = results[0]
    print(f"   📄 Erstes Suchergebnis:")
    print(f"      Inhalt (erste 150 Zeichen): {first_result.page_content[:150]}...")
    print(f"      Metadata: {first_result.metadata}")

def test_search_frankenstein():
    """Test spezifische Suche nach Frankenstein"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
   
    results = vectorstore.similarity_search("Frankenstein monster", k=3)
   
    assert len(results) > 0, "Keine Frankenstein-Ergebnisse gefunden"
   
    # Mindestens ein Ergebnis sollte "frankenstein" im Source haben
    sources = [r.metadata.get('source', '').lower() for r in results]
    frankenstein_found = any('frankenstein' in source for source in sources)
   
    assert frankenstein_found, f"Frankenstein nicht in Quellen gefunden: {sources}"
    print(f"✅ Frankenstein-Suche erfolgreich: {len(results)} Ergebnisse")
    
    # Zeige alle gefundenen Sources
    print(f"   📚 Gefundene Quellen:")
    for i, source in enumerate(sources, 1):
        print(f"      {i}. {source}")

def test_search_quality():
    """Test dass Suchergebnisse relevant sind"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
   
    # Teste verschiedene Queries
    test_queries = [
        "love and romance",
        "adventure and journey",
        "mystery and detective"
    ]
   
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        assert len(results) > 0, f"Keine Ergebnisse für '{query}'"
        assert len(results[0].page_content) > 50, f"Ergebnis zu kurz für '{query}'"
        
        print(f"✅ Query '{query}': {len(results)} Ergebnisse")
        for i, result in enumerate(results, 1):
            source = result.metadata.get('source', 'Unbekannt')
            content_preview = result.page_content[:80] + "..."
            print(f"   {i}. [{source}] {content_preview}")
       
    print("✅ Alle Test-Queries liefern Ergebnisse")

def test_list_all_sources():
    """Listet alle verfügbaren Quellen/Dateien in der Collection auf"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
    
    # Alle Dokumente abrufen (nur Metadaten)
    collection = vectorstore._collection
    
    try:
        # Alle Dokumente mit Metadaten abrufen
        all_data = collection.get(include=['metadatas'])
        
        if all_data and 'metadatas' in all_data:
            # Alle Sources sammeln
            sources = set()
            for metadata in all_data['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
            
            print(f"\n📚 Alle Quellen in der Collection ({len(sources)}):")
            for i, source in enumerate(sorted(sources), 1):
                print(f"   {i}. {source}")
                
            # Dokumente pro Quelle zählen
            print(f"\n📊 Dokumente pro Quelle:")
            source_counts = {}
            for metadata in all_data['metadatas']:
                if metadata and 'source' in metadata:
                    source = metadata['source']
                    source_counts[source] = source_counts.get(source, 0) + 1
            
            for source, count in sorted(source_counts.items()):
                print(f"   {source}: {count} Chunks")
                
        else:
            print("ℹ️ Keine Metadaten gefunden")
            
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Quellen: {e}")

def test_database_statistics():
    """Zeigt ausführliche Statistiken über die Datenbank"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
    collection = vectorstore._collection
    
    print(f"\n📈 Datenbank-Statistiken:")
    print("=" * 40)
    
    try:
        # Grundlegende Info
        count = collection.count()
        print(f"📊 Gesamtanzahl Dokumente: {count}")
        print(f"📚 Collection Name: {collection.name}")
        print(f"🆔 Collection ID: {collection.id}")
        
        if count > 0:
            # Sample von Dokumenten für Analyse
            sample_data = collection.get(limit=min(100, count), include=['documents', 'metadatas'])
            
            if sample_data and 'documents' in sample_data:
                documents = sample_data['documents']
                metadatas = sample_data['metadatas'] or []
                
                # Dokumentlängen analysieren
                doc_lengths = [len(doc) for doc in documents]
                avg_length = sum(doc_lengths) / len(doc_lengths)
                min_length = min(doc_lengths)
                max_length = max(doc_lengths)
                
                print(f"📏 Durchschnittliche Chunk-Länge: {avg_length:.0f} Zeichen")
                print(f"📏 Kürzester Chunk: {min_length} Zeichen")
                print(f"📏 Längster Chunk: {max_length} Zeichen")
                
                # Quellen analysieren
                sources = set()
                for metadata in metadatas:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
                
                print(f"📚 Anzahl verschiedene Quellen: {len(sources)}")
                
                # Embedding-Dimensionen (falls verfügbar)
                try:
                    embeddings_data = collection.get(limit=1, include=['embeddings'])
                    if embeddings_data and 'embeddings' in embeddings_data and embeddings_data['embeddings']:
                        embedding_dim = len(embeddings_data['embeddings'][0])
                        print(f"🧠 Embedding-Dimensionen: {embedding_dim}")
                except:
                    print(f"🧠 Embedding-Dimensionen: Nicht verfügbar")
    
    except Exception as e:
        print(f"❌ Fehler bei Statistik-Erstellung: {e}")