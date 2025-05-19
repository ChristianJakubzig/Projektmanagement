"""
Konversationeller RAG-Chatbot mit Gesprächsverlaufsbewusstsein

Dieses Skript implementiert einen konversationellen Retrieval-Augmented Generation (RAG) Chatbot,
der Gesprächsverlaufsbewusstsein unterstützt. Im Gegensatz zu einfachen RAG-Systemen kann dieser
Chatbot frühere Fragen und Antworten im Kontext berücksichtigen, was natürlichere und kohärentere
Konversationen ermöglicht.

Der Prozess umfasst:
1. Laden einer vorhandenen Vektordatenbank mit Dokumenten
2. Erstellen eines gesprächsverlaufsbewussten Retrievers, der Fragen im Kontext des bisherigen Dialogs versteht
3. Implementierung einer Antwortgenerierung basierend auf abgerufenen Dokumenten
4. Bereitstellung einer interaktiven Chat-Schnittstelle, die den Gesprächsverlauf kontinuierlich aktualisiert

Diese fortgeschrittene RAG-Implementierung ermöglicht es dem Benutzer, auf frühere Teile des
Gesprächs zu verweisen, ohne Kontext erneut angeben zu müssen.
"""

import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells für die Umwandlung von Text in Vektoren
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert, wenn nicht gesetzt
# Name des zu verwendenden Sprachmodells für die Texterzeugung
MODEL_NAME = "llama3.2"  # Hier wird das Llama 3.2 Modell für die Textgenerierung verwendet

# Definition des Verzeichnisses für die persistente Speicherung der Vektordatenbank
# Bestimmt zuerst das aktuelle Verzeichnis des Skripts
current_dir = os.path.dirname(os.path.abspath(__file__))
# Vollständiger Pfad zum Chroma-Datenbankverzeichnis mit Metadaten
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialisierung des Embedding-Modells
# Erstellt ein OllamaEmbeddings-Objekt mit dem konfigurierten Modell und der Server-URL
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Laden der bestehenden Vektordatenbank mit der definierten Embedding-Funktion
# Chroma lädt die Datenbank aus dem angegebenen Verzeichnis und verwendet die Embedding-Funktion
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Erstellen eines Retrievers für die Abfrage der Vektordatenbank
# `search_type` gibt den Typ der Suche an (z.B. Ähnlichkeitssuche)
# `search_kwargs` enthält zusätzliche Argumente für die Suche (z.B. Anzahl der zurückzugebenden Ergebnisse)
retriever = db.as_retriever(
    search_type="similarity",  # Verwendet Ähnlichkeitssuche als Methode
    search_kwargs={"k": 3},    # Ruft die 3 relevantesten Dokumente ab
)

# Erstellen eines ChatOllama-Modells für die Textgenerierung
# Initialisiert das Sprachmodell mit dem konfigurierten Modellnamen und der Server-URL
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

# Systemanweisung zur Kontextualisierung von Fragen
# Diese Anweisung hilft dem KI-Modell zu verstehen, dass es die Frage basierend auf dem
# Gesprächsverlauf umformulieren soll, um eine eigenständige Frage zu erstellen
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Erstellen einer Prompt-Vorlage für die Kontextualisierung von Fragen
# Diese Vorlage kombiniert die Systemanweisung, den Gesprächsverlauf und die aktuelle Benutzereingabe
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),  # Systemanweisung zur Kontextualisierung
        MessagesPlaceholder("chat_history"),        # Platzhalter für den Gesprächsverlauf
        ("human", "{input}"),                       # Platzhalter für die aktuelle Benutzereingabe
    ]
)

# Erstellen eines gesprächsverlaufsbewussten Retrievers
# Dieser verwendet das Sprachmodell, um die Frage basierend auf dem Gesprächsverlauf zu reformulieren
# und so besser zu verstehen, was der Benutzer im aktuellen Kontext meint
history_aware_retriever = create_history_aware_retriever(
    llm,                    # Das Sprachmodell für die Reformulierung
    retriever,              # Der Basis-Retriever für die Dokumentsuche
    contextualize_q_prompt  # Die Prompt-Vorlage für die Kontextualisierung
)

# Systemanweisung zur Beantwortung von Fragen
# Diese Anweisung hilft dem KI-Modell zu verstehen, dass es präzise Antworten
# basierend auf dem abgerufenen Kontext geben soll und gibt an, was zu tun ist,
# wenn die Antwort unbekannt ist
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"  # Platzhalter für den abgerufenen Dokumentkontext
)

# Erstellen einer Prompt-Vorlage für die Beantwortung von Fragen
# Diese Vorlage kombiniert die Systemanweisung, den Gesprächsverlauf und die aktuelle Benutzereingabe
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),        # Systemanweisung zur Beantwortung
        MessagesPlaceholder("chat_history"), # Platzhalter für den Gesprächsverlauf
        ("human", "{input}"),                # Platzhalter für die aktuelle Benutzereingabe
    ]
)

# Erstellen einer Kette zur Kombination von Dokumenten für die Beantwortung von Fragen
# `create_stuff_documents_chain` füttert den gesamten abgerufenen Kontext in das Sprachmodell
# Die "Stuff"-Methode bedeutet, dass alle Dokumente auf einmal in den Prompt gestopft werden
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Erstellen einer Retrieval-Kette, die den gesprächsverlaufsbewussten Retriever und die
# Frage-Antwort-Kette kombiniert. Diese Gesamtkette führt den vollständigen RAG-Prozess durch:
# 1. Kontextualisieren der Frage basierend auf dem Gesprächsverlauf
# 2. Abrufen relevanter Dokumente
# 3. Generieren einer Antwort basierend auf den abgerufenen Dokumenten
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Funktion zur Simulation eines kontinuierlichen Chats
# Diese Funktion ermöglicht eine interaktive Konversation mit dem RAG-System
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Sammelt den Gesprächsverlauf als Sequenz von Nachrichten
    while True:
        # Eingabe vom Benutzer erhalten
        query = input("You: ")
        # Beenden, wenn der Benutzer "exit" eingibt
        if query.lower() == "exit":
            break
        # Verarbeiten der Benutzeranfrage durch die Retrieval-Kette
        # Übergibt die aktuelle Anfrage und den bisherigen Gesprächsverlauf
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Anzeigen der Antwort des KI-Modells
        print(f"AI: {result['answer']}")
        # Aktualisieren des Gesprächsverlaufs mit der aktuellen Frage und Antwort
        # Dies ermöglicht es dem System, auf frühere Teile des Gesprächs zu verweisen
        chat_history.append(HumanMessage(content=query))            # Fügt die Benutzernachricht hinzu
        chat_history.append(SystemMessage(content=result["answer"])) # Fügt die KI-Antwort hinzu


# Hauptfunktion zum Starten des kontinuierlichen Chats
# Wird nur ausgeführt, wenn das Skript direkt (nicht als Modul) ausgeführt wird
if __name__ == "__main__":
    continual_chat()