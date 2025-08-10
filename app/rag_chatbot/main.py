"""
Einfache Main - RAG Chatbot Workflow
"""
from rag_chatbot.embeddings.embedding_manager import EmbeddingManager
from rag_chatbot.embeddings.document_processor import DocumentProcessor
from rag_chatbot.vector_store.chroma_client import get_chroma_vectorstore
from rag_chatbot.vector_store.vector_operations import vector_operations


def main():
    print("🚀 RAG Chatbot starten...")
    
    # 1. Komponenten laden
    embedding_manager = EmbeddingManager()
    processor = DocumentProcessor()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
    vector_ops = vector_operations(vectorstore)
    
    # 2. Dokumente verarbeiten
    documents = processor.process_all_files()
    print(f"📚 {len(documents)} Chunks erstellt")
    
    # 3. Embedden und speichern
    print("⏳ Embedding läuft...")
    vector_ops.add_documents(documents)
    
    # 4. Test-Suche
    results = vectorstore.similarity_search("What is Frankenstein about?", k=2)
    print(f"\n🔍 Test-Suche erfolgreich: {len(results)} Ergebnisse gefunden")
    print(f"📖 Erstes Ergebnis: {results[0].page_content[:100]}...")
    
    print("\n🎉 Fertig! ChromaDB ist bereit.")


if __name__ == "__main__":
    main()