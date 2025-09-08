import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from chroma_client import get_chroma_vectorstore

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
        return "Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer Frage finden.", []
   
    # 3. Kontext aus Dokumenten erstellen
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
   
    # 4. Prompt für LLM erstellen
    prompt = f"""Basierend auf dem folgenden Kontext, beantworte die Frage präzise und hilfreich:
Kontext:
{context}
Frage: {query}
Antwort:"""
   
    # 5. LLM-Antwort generieren
    response = llm.invoke(prompt)
   
    return response.content, relevant_docs

# Streamlit App
def main():
    st.title("🤖 RAG Assistant")
    st.write("Stelle Fragen zu den gespeicherten Dokumenten!")
    
    # Sidebar für Einstellungen
    with st.sidebar:
        st.header("⚙️ Einstellungen")
        k = st.slider("Anzahl Dokumente", min_value=1, max_value=10, value=3)
        score_threshold = st.slider("Relevanz-Schwelle", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat History anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Quellen anzeigen"):
                    for i, doc in enumerate(message["sources"], 1):
                        source = doc.metadata.get('source', 'Unbekannt') if doc.metadata else 'Unbekannt'
                        st.write(f"{i}. **{source}**")
                        st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        st.divider()
    
    # Chat Input
    if prompt := st.chat_input("Deine Frage hier eingeben..."):
        # User Message anzeigen
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Suche nach relevanten Dokumenten..."):
                try:
                    answer, sources = generate_rag_answer(prompt, k=k, score_threshold=score_threshold)
                    st.write(answer)
                    
                    # Quellen anzeigen wenn verfügbar
                    if sources:
                        with st.expander(f"📚 {len(sources)} Quellen gefunden"):
                            for i, doc in enumerate(sources, 1):
                                source = doc.metadata.get('source', 'Unbekannt') if doc.metadata else 'Unbekannt'
                                st.write(f"**{i}. {source}**")
                                st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                                st.divider()
                    
                    # Message zu Session State hinzufügen
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Fehler: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear Chat Button
    if st.button("🗑️ Chat löschen"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()