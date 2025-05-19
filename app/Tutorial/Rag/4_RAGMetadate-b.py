"""
Dokument-Retrieval-Tool für vektorisierte Textdaten

Dieses Skript dient zum Abrufen relevanter Textpassagen aus einer zuvor erstellten Chroma-Vektordatenbank.
Es lädt eine bestehende Vektordatenbank, nutzt das Ollama-Embeddings-Modell, um eine Benutzeranfrage
in einen Vektor umzuwandeln und sucht dann nach den relevantesten Dokumenten in der Datenbank.
Die Ergebnisse werden mit den zugehörigen Metadaten ausgegeben.

Funktionsweise:
1. Konfiguration der Pfade und Modell-Einstellungen
2. Laden der existierenden Chroma-Vektordatenbank
3. Definition einer Beispielanfrage ("Wie starb Julia?")
4. Retrieval der relevantesten Dokumente basierend auf semantischer Ähnlichkeit
5. Ausgabe der gefundenen Textpassagen mit Quellenangaben
"""

import os

# Import der benötigten Bibliothekenmodule
from langchain_community.vectorstores import Chroma  # Vektordatenbank für die Abfrage der gespeicherten Embeddings
from langchain_ollama import OllamaEmbeddings  # Für die Erzeugung von Text-Embeddings mit Ollama

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert

# Definition des Verzeichnisses für die persistente Datenbank
current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Ausführungsverzeichnis des Skripts
db_dir = os.path.join(current_dir, "db")  # Hauptverzeichnis für die Datenbank
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")  # Vollständiger Pfad zum Speicherort der Chroma-Datenbank

# Definition des Embedding-Modells
# Hier wird das gleiche Modell verwendet, das auch zur Erstellung der Vektordatenbank verwendet wurde
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)  # Initialisierung des Embedding-Modells mit den konfigurierten Einstellungen

# Laden der existierenden Vektordatenbank mit der Embedding-Funktion
# Wichtig: Es wird keine neue Datenbank erstellt, sondern eine bestehende geladen
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)  # Verbindung zur vorhandenen Chroma-Datenbank herstellen

# Definition der Benutzeranfrage
# Diese Frage wird in einen Vektor umgewandelt und zum Suchen ähnlicher Dokumente verwendet
query = "How did Juliet die?"  # Beispielfrage: "Wie starb Julia?"

# Abrufen relevanter Dokumente basierend auf der Anfrage
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # Suche basierend auf Ähnlichkeitsschwellenwert
    search_kwargs={"k": 3, "score_threshold": 0.1},  # Parameter: maximal 3 Ergebnisse mit mindestens 0.1 Ähnlichkeitswert
)
relevant_docs = retriever.invoke(query)  # Durchführung der Suche und Speicherung der relevanten Dokumente

# Anzeigen der relevanten Ergebnisse mit Metadaten
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")  # Ausgabe des Dokumenteninhalts
    print(f"Source: {doc.metadata['source']}\n")  # Ausgabe der Quelle aus den Metadaten