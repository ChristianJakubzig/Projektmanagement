import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Streamlit Konfiguration
st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Titel der Anwendung
st.title("🤖 SADPAC")

# Sidebar für Konfiguration
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    # Ollama URL Eingabe
    ollama_url = st.text_input(
        "Ollama URL:", 
        value=os.getenv("OLLAMA_URL", "http://ollama:11434"),
        help="URL des Ollama-Servers"
    )
    
    # Modell Auswahl
    model_name = st.selectbox(
        "Modell auswählen:",
        ["llama3.2", "llama3.1", "mistral", "codellama"],
        index=0,
        help="Wählen Sie das zu verwendende Ollama-Modell"
    )
    
    # System Message anpassen
    system_prompt = st.text_area(
        "System Prompt:",
        value="Du bist ein hilfreicher KI-Assistent. Antworte höflich und informativ auf Deutsch.",
        height=100,
        help="Definiert das Verhalten des Chatbots"
    )
    
    # Chat zurücksetzen Button
    if st.button("🗑️ Chat zurücksetzen", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Session State initialisieren
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model" not in st.session_state or st.session_state.get("current_url") != ollama_url or st.session_state.get("current_model") != model_name:
    try:
        st.session_state.model = ChatOllama(model=model_name, base_url=ollama_url)
        st.session_state.current_url = ollama_url
        st.session_state.current_model = model_name
        st.sidebar.success(f"✅ Verbunden mit {model_name}")
    except Exception as e:
        st.sidebar.error(f"❌ Verbindungsfehler: {str(e)}")
        st.stop()

# Chat History anzeigen
st.subheader("💬 Chat Verlauf")

# Container für Chat Messages
chat_container = st.container()

with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, SystemMessage):
            continue  # System Messages nicht anzeigen
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

# Chat Input
st.subheader("✍️ Neue Nachricht")

# Eingabefeld für neue Nachricht
user_input = st.text_area(
    "Ihre Nachricht:",
    placeholder="Geben Sie hier Ihre Nachricht ein...",
    height=100,
    key="user_input"
)

# Buttons
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    send_button = st.button("📤 Senden", use_container_width=True)

with col2:
    clear_input = st.button("🧹 Leeren", use_container_width=True)

# Input leeren
if clear_input:
    st.session_state.user_input = ""
    st.rerun()

# Nachricht senden und verarbeiten
if send_button and user_input.strip():
    # System Message hinzufügen wenn Chat leer ist
    if not st.session_state.chat_history:
        st.session_state.chat_history.append(SystemMessage(content=system_prompt))
    
    # User Message hinzufügen
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Placeholder für AI Response
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            # Streaming Response
            response_content = ""
            
            # Progress bar für bessere UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🤔 KI denkt nach...")
            
            # Stream die Antwort
            chunk_count = 0
            for chunk in st.session_state.model.stream(st.session_state.chat_history):
                chunk_content = chunk.content
                response_content += chunk_content
                
                # Update der Anzeige
                response_placeholder.write(response_content)
                chunk_count += 1
                
                # Progress bar aktualisieren (simulation)
                if chunk_count % 5 == 0:
                    progress_bar.progress(min(chunk_count * 2, 100))
            
            # Progress bar entfernen
            progress_bar.empty()
            status_text.empty()
            
            # AI Response zur History hinzufügen
            st.session_state.chat_history.append(AIMessage(content=response_content))
            
            # Input field leeren
            st.session_state.user_input = ""
            
        except Exception as e:
            st.error(f"❌ Fehler bei der KI-Antwort: {str(e)}")
    
    # Seite neu laden um aktualisierte History zu zeigen
    st.rerun()

elif send_button and not user_input.strip():
    st.warning("⚠️ Bitte geben Sie eine Nachricht ein.")

# Sidebar Statistiken
with st.sidebar:
    st.header("📊 Statistiken")
    
    # Nachrichtenzähler (ohne System Message)
    user_messages = len([msg for msg in st.session_state.chat_history if isinstance(msg, HumanMessage)])
    ai_messages = len([msg for msg in st.session_state.chat_history if isinstance(msg, AIMessage)])
    
    st.metric("Benutzer Nachrichten", user_messages)
    st.metric("KI Antworten", ai_messages)
    st.metric("Gesamt Nachrichten", user_messages + ai_messages)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: small;'>
    Powered by Streamlit & Ollama | 
    Verwenden Sie den Sidebar um Einstellungen anzupassen
    </div>
    """, 
    unsafe_allow_html=True
)

# Hilfsfunktion für Debug (nur in Development)
if st.sidebar.checkbox("🔧 Debug Modus"):
    st.sidebar.subheader("Debug Informationen")
    st.sidebar.write(f"Aktuelle URL: {ollama_url}")
    st.sidebar.write(f"Aktuelles Modell: {model_name}")
    st.sidebar.write(f"Chat History Länge: {len(st.session_state.chat_history)}")
    
    with st.sidebar.expander("Chat History Details"):
        for i, msg in enumerate(st.session_state.chat_history):
            st.write(f"{i}: {type(msg).__name__}: {msg.content[:50]}...")