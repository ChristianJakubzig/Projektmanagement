"""
Konversationeller RAG-Chatbot mit Gesprächsverlaufsbewusstsein und MultiQueryRetriever

Dieses Skript implementiert einen konversationellen Retrieval-Augmented Generation (RAG) Chatbot,
der Gesprächsverlaufsbewusstsein unterstützt und einen MultiQueryRetriever verwendet.
Der MultiQueryRetriever generiert mehrere Variationen der ursprünglichen Anfrage,
um umfassendere und relevantere Dokumentenabrufe zu ermöglichen.

Der Prozess umfasst:
1. Laden einer vorhandenen Vektordatenbank mit Dokumenten
2. Erstellen eines MultiQueryRetrievers für bessere Dokumentenabrufe
3. Erstellen eines gesprächsverlaufsbewussten Retrievers
4. Implementierung einer Antwortgenerierung basierend auf abgerufenen Dokumenten
5. Bereitstellung einer interaktiven Chat-Schnittstelle

Diese fortgeschrittene RAG-Implementierung ermöglicht präzisere Antworten durch
die Verwendung mehrerer Abfragevariationen.
"""

import os
import logging
import requests

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Logging-Konfiguration für detaillierte Ausgaben des MultiQueryRetrievers
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Konfigurationsvariablen für externe Ollama-Instanz
OLLAMA_URL = "https://ollama-bim24.apps.rhos.th-wildau.de"
MODEL_NAME = "llama3.2"  # Für Chat/Text-Generierung

embeddings = HuggingFaceBgeEmbeddings(model_name="ibm-granite/granite-embedding-278m-multilingual")

# Definition des Verzeichnisses für die persistente Speicherung der Vektordatenbank
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_ollama_external")

def check_ollama_connection():
    """
    Überprüft die Verbindung zur externen Ollama-Instanz
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Verbindung zu Ollama erfolgreich. Verfügbare Modelle: {len(models)}")
            for model in models:
                print(f"  - {model['name']}")
            return True
        else:
            print(f"❌ Ollama nicht erreichbar. Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Fehler bei der Verbindung zu Ollama: {e}")
        return False

def pull_model_if_needed(model_name):
    """
    Lädt ein Modell herunter, falls es nicht vorhanden ist
    """
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

# Verbindung zu Ollama prüfen
print("Prüfe Verbindung zur externen Ollama-Instanz...")
if not check_ollama_connection():
    print("Bitte prüfen Sie die Ollama-URL und Ihre Internetverbindung.")
    exit(1)

# Modelle laden falls nötig
print(f"Prüfe Verfügbarkeit der benötigten Modelle...")
if not pull_model_if_needed(MODEL_NAME):
    print(f"Konnte Modell '{MODEL_NAME}' nicht laden. Verwende verfügbares Modell.")


# Laden oder Erstellen der Vektordatenbank
print("Lade Vektordatenbank...")
try:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    print(f"✅ Vektordatenbank geladen aus: {persistent_directory}")
except Exception as e:
    print(f"❌ Fehler beim Laden der Vektordatenbank: {e}")
    print("Bitte stellen Sie sicher, dass die Datenbank existiert oder erstellen Sie eine neue.")
    exit(1)

# Erstellen des Basis-Retrievers
base_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

# Erstellen des ChatOllama-Modells mit externer Instanz
llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    # Ressourcenschonende Einstellungen für geteilte GPU
    num_ctx=4096,  # Kontextlänge begrenzen
    temperature=0.7,
    # Modell nach 5 Minuten Inaktivität entladen (Standard)
    keep_alive="5m"
)

# Erstellen des MultiQueryRetrievers
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    parser_key="lines"
)

# Verbesserter Prompt für deutsche Abfragegenerierung
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sie sind ein KI-Assistent für die Generierung von Suchanfragen. 
    Ihre Aufgabe ist es, basierend auf der Benutzerfrage 5 verschiedene, aber ähnliche Suchvarianten zu erstellen, 
    um relevante Dokumente aus einer Vektordatenbank zu finden. 
    
    Die Varianten sollten:
    - Synonyme und alternative Begriffe verwenden
    - Verschiedene Formulierungen derselben Frage darstellen
    - Den gleichen Informationsbedarf abdecken
    - Auf Deutsch formuliert sein
    
    Geben Sie die 5 Varianten als separate Zeilen aus, ohne Nummerierung.
    
    Ursprüngliche Frage: {question}"""),
])

# MultiQueryRetriever mit benutzerdefiniertem Prompt
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=custom_prompt
)

# Systemanweisung zur Kontextualisierung von Fragen
contextualize_q_system_prompt = (
    "Basierend auf einem Chat-Verlauf und der neuesten Benutzerfrage, "
    "die sich auf den Kontext im Chat-Verlauf beziehen könnte, "
    "formulieren Sie eine eigenständige Frage, die ohne den Chat-Verlauf "
    "verstanden werden kann. Beantworten Sie die Frage NICHT, sondern "
    "formulieren Sie sie nur bei Bedarf um, andernfalls geben Sie sie so zurück, wie sie ist."
)

# Prompt-Vorlage für die Kontextualisierung
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Erstellen eines gesprächsverlaufsbewussten Retrievers
history_aware_retriever = create_history_aware_retriever(
    llm,
    multi_query_retriever,
    contextualize_q_prompt
)

# Systemanweisung zur Beantwortung von Fragen
qa_system_prompt = (
    "Sie sind ein hilfsreicher KI-Assistent für Frage-Antwort-Aufgaben. "
    "Verwenden Sie die folgenden abgerufenen Kontextinformationen, um die Frage präzise zu beantworten. "
    "Wenn Sie die Antwort nicht aus dem Kontext ableiten können, sagen Sie ehrlich, dass Sie es nicht wissen. "
    "Halten Sie Ihre Antworten prägnant aber informativ. "
    "Antworten Sie immer auf Deutsch."
    "\n\n"
    "Kontext:\n{context}"
)

# Prompt-Vorlage für die Beantwortung
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Erstellen der Dokumentenkombinations-Kette
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Erstellen der vollständigen RAG-Kette
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    """
    Funktion zur Simulation eines kontinuierlichen Chats mit externer Ollama-Instanz
    """
    print("🤖 RAG-Chatbot mit externer Ollama-Instanz gestartet!")
    print("💡 Geben Sie 'exit' ein, um das Gespräch zu beenden.")
    print("🔍 MultiQueryRetriever ist aktiv für bessere Suchergebnisse.")
    print("⚡ Modelle werden nach 5 Minuten Inaktivität automatisch entladen.")
    print("-" * 70)
    
    chat_history = []
    
    while True:
        try:
            # Eingabe vom Benutzer erhalten
            query = input("Sie: ")
            
            # Beenden, wenn der Benutzer "exit" eingibt
            if query.lower() in ['exit', 'quit', 'bye']:
                print("👋 Auf Wiedersehen!")
                break
            
            if not query.strip():
                continue
            
            # Verarbeiten der Benutzeranfrage durch die RAG-Kette
            print("🔍 Suche nach relevanten Informationen...")
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            # Anzeigen der Antwort
            print(f"🤖 KI: {result['answer']}")
            
            # Optionale Anzeige der Quellen
            if 'context' in result and result['context']:
                print(f"📚 Basierend auf {len(result['context'])} Dokumenten")
            
            print("-" * 70)
            
            # Aktualisieren des Gesprächsverlaufs
            chat_history.append(HumanMessage(content=query))
            chat_history.append(SystemMessage(content=result["answer"]))
            
        except KeyboardInterrupt:
            print("\n👋 Chat beendet.")
            break
        except Exception as e:
            print(f"❌ Fehler bei der Verarbeitung: {e}")
            print("🔄 Bitte versuchen Sie es erneut.")


def show_multi_query_details(query):
    """
    Hilfsfunktion zur Demonstration der MultiQuery-Funktionalität
    """
    print(f"🔍 Ursprüngliche Anfrage: {query}")
    print("📝 Analysiere und generiere Abfragevariationen...")
    
    try:
        docs = multi_query_retriever.get_relevant_documents(query)
        print(f"📄 Gefundene Dokumente: {len(docs)}")
        
        for i, doc in enumerate(docs[:3]):  # Zeige nur die ersten 3
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"   {i+1}. {preview}...")
            
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Details: {e}")


# Hauptfunktion
if __name__ == "__main__":
    print("🚀 RAG-Chatbot mit externer Ollama-Instanz")
    print("🌐 Ollama-Server: https://ollama-bim24.apps.rhos.th-wildau.de")
    print("=" * 70)
    
    # Optional: Demonstration der MultiQuery-Funktionalität
    demo_mode = input("💡 Möchten Sie eine Demo der MultiQuery-Funktionalität sehen? (j/n): ")
    if demo_mode.lower() in ['j', 'ja', 'y', 'yes']:
        demo_query = input("📝 Geben Sie eine Beispielfrage ein: ")
        show_multi_query_details(demo_query)
        print("=" * 70)
    
    # Starten des Chats
    continual_chat()