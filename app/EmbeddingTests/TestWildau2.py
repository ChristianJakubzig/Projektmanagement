"""
Streamlit GUI für RAG-Chatbot mit Streaming

Dieses Skript erstellt eine benutzerfreundliche Web-Oberfläche für den RAG-Chatbot
mit Streaming-Antworten und Standard-Retriever.

Angepasst für TH-Wildau Ollama-Cluster:
- Neue Ollama-URL: https://ollama-bim24.apps.rhos.th-wildau.de
- OpenAI-kompatible API für bessere Performance
- Ressourcenschonendes Keep-Alive Management
- Beibehaltung des lokalen HuggingFace Embedding-Modells

Features:
- Moderne Streamlit-Benutzeroberfläche
- Streaming von KI-Antworten in Echtzeit
- Chat-Verlauf mit Benutzer- und KI-Nachrichten
- Standard-Retriever für Suchergebnisse
- Seitenleiste mit Konfigurationsoptionen
- Zurücksetzen des Chat-Verlaufs
- Anzeige der Quelldokumente
"""

import os
import logging
import streamlit as st
import requests
from typing import Dict, Any, Generator

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama  # Zurück zu ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.callbacks import BaseCallbackHandler

# Streamlit Konfiguration
st.set_page_config(
    page_title="RAG-Chatbot (TH-Wildau)",
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
    .cluster-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
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

def check_ollama_connection():
    """
    Überprüft die Verbindung zur externen Ollama-Instanz
    """
    OLLAMA_URL = "https://ollama-bim24.apps.rhos.th-wildau.de"
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Verbindung zu Ollama erfolgreich. Verfügbare Modelle: {len(models)}")
            for model in models:
                print(f"  - {model['name']}")
            return True, models
        else:
            print(f"❌ Ollama nicht erreichbar. Status Code: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Fehler bei der Verbindung zu Ollama: {e}")
        return False, []

def pull_model_if_needed(model_name):
    """
    Lädt ein Modell herunter, falls es nicht vorhanden ist
    """
    OLLAMA_URL = "https://ollama-bim24.apps.rhos.th-wildau.de"
    try:
        # Prüfen ob Modell verfügbar ist
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            available_models = [model['name'] for model in response.json().get('models', [])]
            if model_name in available_models:
                print(f"✅ Modell '{model_name}' ist bereits verfügbar")
                return True
       
        print(f"📥 Lade Modell '{model_name}' herunter...")
        pull_response = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model_name},
            timeout=300  # 5 Minuten Timeout für Download
        )
       
        if pull_response.status_code == 200:
            print(f"✅ Modell '{model_name}' erfolgreich geladen")
            return True
        else:
            print(f"❌ Fehler beim Laden des Modells '{model_name}'")
            return False
           
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        return False

@st.cache_resource
def initialize_rag_system():
    """
    Initialisiert das RAG-System und cached es für bessere Performance
    Angepasst für TH-Wildau Ollama-Cluster mit llama3.2
    """
    # Konfigurationsvariablen für TH-Wildau Cluster
    OLLAMA_URL = "https://ollama-bim24.apps.rhos.th-wildau.de"
    MODEL_NAME = "llama3.2"  # Geändert zu llama3.2
    
    # Verbindung zu Ollama prüfen
    print("Prüfe Verbindung zur externen Ollama-Instanz...")
    connection_ok, available_models = check_ollama_connection()
    if not connection_ok:
        st.error("❌ Keine Verbindung zum TH-Wildau Ollama-Cluster möglich")
        st.stop()
    
    # Modelle laden falls nötig
    print(f"Prüfe Verfügbarkeit der benötigten Modelle...")
    if not pull_model_if_needed(MODEL_NAME):
        st.warning(f"⚠️ Konnte Modell '{MODEL_NAME}' nicht laden. Verwende verfügbares Modell.")
        # Fallback zu verfügbarem Modell falls nötig
        if available_models:
            MODEL_NAME = available_models[0]['name']
            st.info(f"📋 Verwende Fallback-Modell: {MODEL_NAME}")
    
    # Pfad zur Vektordatenbank
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_huggingface_custom2")
    
    # Überprüfen ob die Datenbank existiert
    if not os.path.exists(persistent_directory):
        st.error(f"Vektordatenbank nicht gefunden: {persistent_directory}")
        st.stop()
    
    try:
        # Embedding-Modell initialisieren (bleibt lokal)
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="oliverguhr/revosax-granite-embedding-278m-multilingual"
        )
        
        # Vektordatenbank laden
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        
        # Standard-Retriever erstellen (ohne MultiQuery)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        
        # LLM initialisieren mit optimierter Konfiguration
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_URL,
            # Ressourcenschonende Einstellungen für geteilte GPU
            num_ctx=4096,  # Kontextlänge begrenzen
            temperature=0.7,
            # Modell nach 5 Minuten Inaktivität entladen (Standard)
            keep_alive="5m"
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
            "Sie sind ein präziser Assistent für Frage-Antwort-Aufgaben. "
            "Beantworten Sie die Frage ausschließlich mit den folgenden abgerufenen Textabschnitten (Kontext). "
            "Fügen Sie keine zusätzlichen Details, Spekulationen oder Informationen hinzu, die nicht explizit in den Textabschnitten enthalten sind, wie z. B. Giftpilze, Nightshade, Mohnblumen oder falsche Behauptungen über Handlungsdetails. "
            "Wenn die Antwort nicht vollständig in den Textabschnitten enthalten ist, geben Sie an, dass die Information fehlt, und formulieren Sie die Antwort so genau wie möglich basierend auf den verfügbaren Texten. "
            "Antworten Sie direkt, präzise und auf Deutsch, ohne spekulative Überlegungen oder zusätzliche Interpretationen.\n\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # RAG-Chain erstellen (mit Standard-Retriever)
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain, llm, OLLAMA_URL, MODEL_NAME
        
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
    st.title("🤖 SADPAC (TH-Wildau Cluster)")
    st.markdown("Stellen Sie Fragen basierend auf Ihren Dokumenten!")
    
    # Cluster-Info Banner
    st.markdown("""
    <div class="cluster-info">
        <strong>🖥️ TH-Wildau KI-Cluster:</strong> Dieses System nutzt den Ollama-Service 
        des TH-Wildau Clusters. Bitte nutzen Sie die Ressourcen verantwortungsbewusst.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar für Konfiguration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # Reset Button
        if st.button("🗑️ Chat zurücksetzen", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        # Cluster-Informationen
        st.header("🖥️ Cluster-Info")
        st.markdown("""
        **TH-Wildau KI-Cluster:**
        - 🌐 Ollama-URL: `ollama-bim24.apps.rhos.th-wildau.de`
        - 🤖 Modell: `llama3.2`
        - 💾 GPU-VRAM: 80 GB (geteilt)
        - 🧠 Kontext: 4096 Token
        - ⏱️ Keep-Alive: 5 Minuten
        - 🔄 Auto-Entladung: Aktiviert
        
        **Ressourcenschonung:**
        - Modelle werden nach 5 Min. entladen
        - Geteilte GPU-Nutzung
        - Lokales Embedding-Modell
        """)
        
        st.divider()
        
        # Informationen
        st.header("ℹ️ Features")
        st.markdown("""
        **Funktionen:**
        - 🔍 Standard-Retriever
        - 💬 Chat-Verlauf
        - 📚 Quelldokumente
        - ⚡ Streaming-Antworten
        - 🔌 Verbindungsprüfung
        - 📥 Auto-Modelldownload
        
        **Verwendung:**
        Stellen Sie Ihre Fragen im Chat-Bereich. 
        Der Bot durchsucht Ihre Dokumente und 
        antwortet basierend auf den gefundenen Informationen.
        """)
        
        # Status
        st.header("📊 Status")
        try:
            rag_chain, llm, ollama_url, model_name = initialize_rag_system()
            st.success("✅ RAG-System bereit")
            st.success("✅ TH-Wildau Cluster verbunden")
            st.info(f"🔗 {ollama_url}")
            st.info(f"🤖 Aktives Modell: {model_name}")
        except Exception as e:
            st.error("❌ RAG-System nicht verfügbar")
            st.error(f"Fehler: {str(e)}")
    
    # Chat-System initialisieren
    try:
        rag_chain, llm, ollama_url, model_name = initialize_rag_system()
    except:
        st.error("Das RAG-System konnte nicht initialisiert werden. Bitte überprüfen Sie die Verbindung zum TH-Wildau Cluster.")
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
                with st.spinner("🔍 Suche nach relevanten Informationen im TH-Wildau Cluster..."):
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
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    error_msg += "\n\n💡 Mögliche Ursachen:\n- TH-Wildau Cluster nicht erreichbar\n- Netzwerkverbindung unterbrochen\n- Modell wird gerade geladen"
                
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

if __name__ == "__main__":
    main()