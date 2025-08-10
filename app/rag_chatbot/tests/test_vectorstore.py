"""
Test für die Vectorstore Funktionalität
"""
from rag_chatbot.embeddings.embedding_manager import EmbeddingManager
from rag_chatbot.vector_store.chroma_client import get_chroma_vectorstore


def test_vectorstore_has_data():
    """Test dass die Datenbank Daten enthält"""
    embedding_manager = EmbeddingManager()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
    
    # Test-Suche
    results = vectorstore.similarity_search("test", k=1)
    
    # Assertions
    assert len(results) > 0, "Datenbank ist leer - setup_database.py ausführen!"
    assert results[0].page_content, "Ergebnis hat keinen Inhalt"
    print(f"✅ Datenbank enthält {len(results)} Ergebnisse")


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
        
    print("✅ Alle Test-Queries liefern Ergebnisse")