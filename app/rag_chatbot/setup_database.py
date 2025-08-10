"""
setup_database.py - Einmalig ausführen zum Embedden aller Bücher
"""
from rag_chatbot.embeddings.embedding_manager import EmbeddingManager
from rag_chatbot.embeddings.document_processor import DocumentProcessor
from rag_chatbot.vector_store.chroma_client import get_chroma_vectorstore
from rag_chatbot.vector_store.vector_operations import vector_operations


def setup_database():
    print("📚 Datenbank wird einmalig erstellt...")
    print("⚠️  Das kann ein paar Minuten dauern!")
    print("=" * 50)
    
    # 1. Komponenten laden
    print("🔧 Komponenten laden...")
    embedding_manager = EmbeddingManager()
    processor = DocumentProcessor()
    vectorstore = get_chroma_vectorstore(embedding_manager.get_model())
    vector_ops = vector_operations(vectorstore)
    
    # 2. Alle Bücher verarbeiten
    print("📖 Bücher verarbeiten...")
    documents = processor.process_all_files()
    
    # 3. Embedden und speichern
    print("🧠 Embedding und Speichern...")
    vector_ops.add_documents(documents)
    
    print("\n🎉 Setup abgeschlossen!")
    print("💡 Teste mit: pytest rag_chatbot/tests/test_vectorstore.py")


if __name__ == "__main__":
    setup_database()