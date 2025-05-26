"""
Streamlit GUI für RAG-Chatbot mit MultiQueryRetriever und Streaming

Dieses Skript erstellt eine benutzerfreundliche Web-Oberfläche für den RAG-Chatbot
mit Streaming-Antworten und erweiterten Features.

Features:
- Moderne Streamlit-Benutzeroberfläche
- Streaming von KI-Antworten in Echtzeit
- Chat-Verlauf mit Benutzer- und KI-Nachrichten
- MultiQueryRetriever für bessere Suchergebnisse
- Seitenleiste mit Konfigurationsoptionen
- Zurücksetzen des Chat-Verlaufs
- Anzeige der Quelldokumente
"""

import os
import logging
import streamlit as st
from typing import Dict, Any, Generator

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import BaseCallbackHandler

# Streamlit Konfiguration
st.set_page_config(
    page_title="RAG-Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für besseres Design
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-docs {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Callback Handler für Streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

@st.cache_resource
def initialize_rag_system():
    """
    Initialisiert das RAG-System und cached es für bessere Performance
    """
    # Konfigurationsvariablen
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    MODEL_NAME = "llama3.2"
    
    # Pfad zur Vektordatenbank
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_huggingface_custom")
    
    # Überprüfen ob die Datenbank existiert
    if not os.path.exists(persistent_directory):
        st.error(f"Vektordatenbank nicht gefunden: {persistent_directory}")
        st.stop()
    
    try:
        # Embedding-Modell initialisieren
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="ibm-granite/granite-embedding-278m-multilingual"
        )
        
        # Vektordatenbank laden
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        
        # Basis-Retriever erstellen
        base_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
        )
        
        # LLM initialisieren
        llm = ChatOllama(
            model=MODEL_NAME, 
            base_url=OLLAMA_URL,
            temperature=0.1,
            streaming=True  # Aktiviert Streaming
        )
        
        # MultiQueryRetriever erstellen
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            parser_key="lines"
        )
        
        # Prompts definieren
        contextualize_q_system_prompt = (
            "Basierend auf einem Chat-Verlauf und der neuesten Benutzerfrage, "
            "die sich auf den Kontext im Chat-Verlauf beziehen könnte, "
            "formulieren Sie eine eigenständige Frage, die ohne den Chat-Verlauf "
            "verstanden werden kann. Beantworten Sie die Frage NICHT, sondern "
            "formulieren Sie sie nur bei Bedarf um, andernfalls geben Sie sie so zurück, wie sie ist."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        qa_system_prompt = (
            "Sie sind ein hilfsreicher Assistent für Frage-Antwort-Aufgaben. "
            "Verwenden Sie die folgenden abgerufenen Kontextinformationen, um die Frage zu beantworten. "
            "Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen. "
            "Antworten Sie präzise und auf Deutsch.\n\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # RAG-Chain erstellen
        history_aware_retriever = create_history_aware_retriever(
            llm, multi_query_retriever, contextualize_q_prompt
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain, llm
        
    except Exception as e:
        st.error(f"Fehler bei der Initialisierung des RAG-Systems: {e}")
        st.stop()

def stream_response(rag_chain, query: str, chat_history: list) -> Generator[str, None, None]:
    """
    Streamt die Antwort des RAG-Systems
    """
    try:
        # Invoke the chain
        result = rag_chain.invoke({
            "input": query, 
            "chat_history": chat_history
        })
        
        # Simuliere Streaming durch Wort-für-Wort Ausgabe
        answer = result.get("answer", "Entschuldigung, ich konnte keine Antwort generieren.")
        words = answer.split()
        
        partial_answer = ""
        for word in words:
            partial_answer += word + " "
            yield partial_answer.strip()
            
        return result
        
    except Exception as e:
        yield f"Fehler bei der Verarbeitung: {str(e)}"

def display_source_documents(context_docs):
    """
    Zeigt die Quelldokumente in einem erweiterbaren Bereich an
    """
    if context_docs:
        with st.expander(f"📚 Quelldokumente ({len(context_docs)} gefunden)", expanded=False):
            for i, doc in enumerate(context_docs):
                st.markdown(f"**Dokument {i+1}:**")
                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.json(doc.metadata)
                st.divider()

def main():
    # Titel und Beschreibung
    st.title("🤖 RAG-Chatbot mit MultiQuery")
    st.markdown("Stellen Sie Fragen basierend auf Ihren Dokumenten - mit verbesserter Suche!")
    
    # Sidebar für Konfiguration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # Reset Button
        if st.button("🗑️ Chat zurücksetzen", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        # Informationen
        st.header("ℹ️ Informationen")
        st.markdown("""
        **Features:**
        - 🔍 MultiQuery-Retriever
        - 💬 Chat-Verlauf
        - 📚 Quelldokumente
        - ⚡ Streaming-Antworten
        
        **Verwendung:**
        Stellen Sie Ihre Fragen im Chat-Bereich. 
        Der Bot durchsucht Ihre Dokumente und 
        antwortet basierend auf den gefundenen Informationen.
        """)
        
        # Status
        st.header("📊 Status")
        try:
            rag_chain, llm = initialize_rag_system()
            st.success("✅ RAG-System bereit")
        except:
            st.error("❌ RAG-System nicht verfügbar")
    
    # Chat-System initialisieren
    try:
        rag_chain, llm = initialize_rag_system()
    except:
        st.error("Das RAG-System konnte nicht initialisiert werden. Bitte überprüfen Sie die Konfiguration.")
        return
    
    # Session State für Chat-Verlauf
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat-Verlauf anzeigen
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Quelldokumente anzeigen falls vorhanden
                if "context" in message:
                    display_source_documents(message["context"])
    
    # Chat Input
    if prompt := st.chat_input("Stellen Sie Ihre Frage hier..."):
        # Benutzernachricht hinzufügen
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Benutzernachricht anzeigen
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant Antwort
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # RAG-Chain ausführen
                with st.spinner("🔍 Suche nach relevanten Informationen..."):
                    result = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                
                # Antwort extrahieren
                answer = result.get("answer", "Entschuldigung, ich konnte keine Antwort generieren.")
                context_docs = result.get("context", [])
                
                # Streaming-Simulation
                displayed_text = ""
                words = answer.split()
                
                for i, word in enumerate(words):
                    displayed_text += word + " "
                    message_placeholder.markdown(displayed_text + "▌")
                    # Kleine Verzögerung für Streaming-Effekt
                    import time
                    time.sleep(0.05)
                
                # Finale Antwort ohne Cursor
                message_placeholder.markdown(displayed_text.strip())
                
                # Quelldokumente anzeigen
                if context_docs:
                    display_source_documents(context_docs)
                
                # Nachrichten zum Session State hinzufügen
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": context_docs
                })
                
                # Chat-Verlauf für nächste Anfrage aktualisieren
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(SystemMessage(content=answer))
                
            except Exception as e:
                error_msg = f"❌ Fehler bei der Verarbeitung: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

if __name__ == "__main__":
    main()