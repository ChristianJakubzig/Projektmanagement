"""
Retrievalgestützte Generierung mit LangChain und Ollama

Dieses Skript demonstriert einen einfachen Retrieval-Augmented Generation (RAG) Prozess
unter Verwendung von LangChain, Chroma als Vektordatenbank und Ollama für Embeddings 
und Sprachmodellierung. Der Prozess umfasst das Abrufen relevanter Dokumente aus einer 
Vektordatenbank basierend auf einer Benutzeranfrage und die anschließende Verwendung 
dieser Dokumente zur Anreicherung einer Anfrage an ein Sprachmodell, um eine fundierte 
Antwort zu generieren.

Der Ablauf ist:
1. Laden einer vorhandenen Vektordatenbank
2. Abfragen relevanter Dokumente basierend auf der Benutzeranfrage
3. Kombinieren der Anfrage mit den gefundenen Dokumenten
4. Senden der angereicherten Anfrage an ein Sprachmodell
5. Anzeigen der generierten Antwort
"""

import os

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
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
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Initialisierung des Embedding-Modells
# Erstellt ein OllamaEmbeddings-Objekt mit dem konfigurierten Modell und der Server-URL
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Laden der bestehenden Vektordatenbank mit der definierten Embedding-Funktion
# Chroma lädt die Datenbank aus dem angegebenen Verzeichnis und verwendet die Embedding-Funktion
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Definition der Benutzeranfrage
query = "How can I learn more about LangChain?"  # Beispielanfrage zur LangChain-Lernressourcen

# Abrufen relevanter Dokumente basierend auf der Anfrage
# Erstellt einen Retriever mit dem Suchtyp "similarity" (Ähnlichkeitssuche)
retriever = db.as_retriever(
    search_type="similarity",  # Verwendet Ähnlichkeitssuche als Methode
    search_kwargs={"k": 1},    # Ruft nur das relevanteste Dokument ab
)
# Führt die Suche durch und speichert die gefundenen Dokumente
relevant_docs = retriever.invoke(query)

# Anzeigen der relevanten Ergebnisse mit Metadaten
print("\n--- Relevante Dokumente ---")
for i, doc in enumerate(relevant_docs, 1):
    # Gibt den Inhalt jedes gefundenen Dokuments aus
    print(f"Dokument {i}:\n{doc.page_content}\n")

# Kombinieren der Anfrage und der relevanten Dokumentinhalte
# Erstellt eine angereicherte Eingabe für das Sprachmodell durch Zusammenführen von:
# 1. Einer Anweisung für das Modell
# 2. Der ursprünglichen Benutzeranfrage
# 3. Den Inhalten der gefundenen relevanten Dokumente
# 4. Einer zusätzlichen Anweisung zur Beantwortung basierend auf den Dokumenten
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Erstellen eines ChatOllama-Modells für die Textgenerierung
# Initialisiert das Sprachmodell mit dem konfigurierten Modellnamen und der Server-URL
model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

# Definition der Nachrichten für das Modell
# Erstellt eine Liste von Nachrichten für das Chat-Modell:
# 1. Eine Systemnachricht, die die Rolle des Modells definiert
# 2. Eine Benutzernachricht, die die kombinierte Eingabe enthält
messages = [
    SystemMessage(content="You are a helpful assistant."),  # Definiert die Rolle des Assistenten
    HumanMessage(content=combined_input),  # Enthält die angereicherte Anfrage mit Dokumenten
]

# Aufrufen des Modells mit der kombinierten Eingabe
# Sendet die Nachrichten an das Sprachmodell und erhält eine Antwort
result = model.invoke(messages)

# Anzeigen des vollständigen Ergebnisses und nur des Inhalts
print("\n--- Generierte Antwort ---")
# Die Ausgabe des vollständigen Ergebnisses ist auskommentiert
# print("Vollständiges Ergebnis:")
# print(result)
print("Nur Inhalt:")
print(result.content)  # Gibt nur den Textinhalt der Antwort aus