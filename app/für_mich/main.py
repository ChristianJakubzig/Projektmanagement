from langchain_ollama import OllamaEmbeddings, ChatOllama
from chroma_client import get_chroma_vectorstore
from config import Config

def generate_rag_answer(query, k=3, score_threshold=0.3):
    """
    Generiert eine Antwort mit RAG (Retrieval-Augmented Generation)
    """
    # 1. Modelle initialisieren
    emb = OllamaEmbeddings(
        base_url="https://ollama-bim24.apps.rhos.th-wildau.de",
        model="granite-embedding:278m",
        keep_alive=300,
    )
    
    llm = ChatOllama(
        base_url="https://ollama-bim24.apps.rhos.th-wildau.de",
        model="llama3.2",
        keep_alive="5m",
        temperature=0.7,
    )
    
    db = get_chroma_vectorstore(emb)
    
    # 2. Relevante Dokumente abrufen
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        print("❌ Keine relevanten Dokumente gefunden.")
        return "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer Frage finden."
    
    # 3. Kontext aus Dokumenten erstellen
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 4. Prompt für LLM erstellen
    prompt = f"""Basierend auf dem folgenden Kontext, beantworte die Frage präzise und hilfreich:

Kontext:
{context}

Frage: {query}

Antwort:"""
    
    # 5. LLM-Antwort generieren
    print("🔍 Relevante Dokumente gefunden:", len(relevant_docs))
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get('source', 'Unbekannt') if doc.metadata else 'Unbekannt'
        print(f"   {i}. {source}")
    
    print("\n🤖 Generiere Antwort...")
    response = llm.invoke(prompt)
    
    return response.content

def main():
    """
    Hauptfunktion für RAG-Chat
    """
    print("🚀 RAG Assistant gestartet!")
    print("Stelle Fragen zu den gespeicherten Dokumenten.")
    print("Tippe 'quit' zum Beenden.\n")
    
    while True:
        query = input("\n❓ Deine Frage: ").strip()
        
        if query.lower() in ['quit', 'exit', 'bye']:
            print("👋 Auf Wiedersehen!")
            break
        
        if not query:
            continue
        
        try:
            answer = generate_rag_answer(query)
            print(f"\n🤖 Antwort:\n{answer}")
        except Exception as e:
            print(f"❌ Fehler: {e}")

# Beispiel-Aufruf
if __name__ == "__main__":
    # Direkter Test
    test_query = "Tell me how does the first character is called that captain Ahab meets in moby dick?"
    print("=== TEST ===")
    answer = generate_rag_answer(test_query)
    print(f"Frage: {test_query}")
    print(f"Antwort: {answer}")
    
    # Interaktiver Modus
    # main()