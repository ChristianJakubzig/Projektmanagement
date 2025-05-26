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

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever

# Logging-Konfiguration für detaillierte Ausgaben des MultiQueryRetrievers
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Konfigurationsvariablen
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
# Name des zu verwendenden Sprachmodells für die Texterzeugung
MODEL_NAME = "llama3.2"

# Definition des Verzeichnisses für die persistente Speicherung der Vektordatenbank
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_huggingface_custom")

# Initialisierung des Embedding-Modells
embeddings = HuggingFaceBgeEmbeddings(model_name="ibm-granite/granite-embedding-278m-multilingual")

# Laden der bestehenden Vektordatenbank
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Erstellen des Basis-Retrievers
base_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

# Erstellen des ChatOllama-Modells
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

# Erstellen des MultiQueryRetrievers
# Dieser generiert automatisch mehrere Variationen der ursprünglichen Anfrage
# um umfassendere Suchergebnisse zu erhalten
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,  # Der Basis-Retriever
    llm=llm,                   # Das LLM zur Generierung der Abfragevariationen
    parser_key="lines"         # Parsing-Methode für die generierten Abfragen
)

# Alternative: Benutzerdefinierter Prompt für den MultiQueryRetriever
# Falls Sie spezifische Anweisungen für die Abfragegenerierung benötigen
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """Sie sind ein KI-Sprachmodell-Assistent. Ihre Aufgabe ist es, fünf verschiedene Versionen der gegebenen Benutzerfrage zu generieren, um relevante Dokumente aus einer Vektordatenbank abzurufen. Durch die Bereitstellung mehrerer Perspektiven auf die Benutzerfrage können Sie dem Benutzer helfen, einige der Einschränkungen der entfernungsbasierten Ähnlichkeitssuche zu überwinden. Stellen Sie diese alternativen Fragen durch Zeilenumbrüche getrennt bereit.

Ursprüngliche Frage: {question}"""),
])

# MultiQueryRetriever mit benutzerdefiniertem Prompt (optional)
# multi_query_retriever = MultiQueryRetriever.from_llm(
#     retriever=base_retriever,
#     llm=llm,
#     prompt=custom_prompt
# )

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

# Erstellen eines gesprächsverlaufsbewussten Retrievers mit dem MultiQueryRetriever
# Dieser kombiniert die Vorteile des Gesprächsverlaufsbewusstseins mit
# der erweiterten Abfragefähigkeit des MultiQueryRetrievers
history_aware_retriever = create_history_aware_retriever(
    llm,
    multi_query_retriever,  # Verwendet den MultiQueryRetriever anstelle des einfachen Retrievers
    contextualize_q_prompt
)

# Systemanweisung zur Beantwortung von Fragen
qa_system_prompt = (
    "Sie sind ein Assistent für Frage-Antwort-Aufgaben. Verwenden Sie "
    "die folgenden abgerufenen Kontextinformationen, um die Frage zu beantworten. "
    "Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen. "
    "Verwenden Sie maximal drei Sätze und halten Sie die Antwort prägnant. "
    "Antworten Sie auf Deutsch."
    "\n\n"
    "{context}"
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

# Erstellen der vollständigen RAG-Kette mit MultiQueryRetriever
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    """
    Funktion zur Simulation eines kontinuierlichen Chats mit verbessertem MultiQuery-Retrieval
    """
    print("Starten Sie den Chat mit der KI! Geben Sie 'exit' ein, um das Gespräch zu beenden.")
    print("Der Chatbot verwendet jetzt MultiQueryRetriever für bessere Suchergebnisse.")
    print("-" * 70)
    
    chat_history = []
    
    while True:
        # Eingabe vom Benutzer erhalten
        query = input("Sie: ")
        
        # Beenden, wenn der Benutzer "exit" eingibt
        if query.lower() == "exit":
            print("Auf Wiedersehen!")
            break
        
        try:
            # Verarbeiten der Benutzeranfrage durch die RAG-Kette
            print("🔍 Suche nach relevanten Informationen...")
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            # Anzeigen der Antwort
            print(f"KI: {result['answer']}")
            print("-" * 70)
            
            # Aktualisieren des Gesprächsverlaufs
            chat_history.append(HumanMessage(content=query))
            chat_history.append(SystemMessage(content=result["answer"]))
            
        except Exception as e:
            print(f"Fehler bei der Verarbeitung Ihrer Anfrage: {e}")
            print("Bitte versuchen Sie es erneut.")


def show_multi_query_details(query):
    """
    Hilfsfunktion zur Demonstration der MultiQuery-Funktionalität
    Zeigt die generierten Abfragevariationen an
    """
    print(f"Ursprüngliche Anfrage: {query}")
    print("Generierte Abfragevariationen:")
    
    try:
        # Direkte Verwendung des MultiQueryRetrievers um die Variationen zu sehen
        docs = multi_query_retriever.get_relevant_documents(query)
        print(f"Gefundene Dokumente: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"Dokument {i+1}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Fehler beim Abrufen der Details: {e}")


# Hauptfunktion
if __name__ == "__main__":
    print("RAG-Chatbot mit MultiQueryRetriever gestartet")
    print("=" * 70)
    
    # Optional: Demonstration der MultiQuery-Funktionalität
    demo_mode = input("Möchten Sie eine Demo der MultiQuery-Funktionalität sehen? (j/n): ")
    if demo_mode.lower() in ['j', 'ja', 'y', 'yes']:
        demo_query = input("Geben Sie eine Beispielfrage ein: ")
        show_multi_query_details(demo_query)
        print("=" * 70)
    
    # Starten des Chats
    continual_chat()