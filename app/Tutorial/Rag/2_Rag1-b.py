"""
Semantische Suche in einer Textdatenbank mit LangChain und Ollama

Dieses Skript führt eine semantische Suche in einer vorhandenen Vektordatenbank durch.
Es lädt eine zuvor erstellte Chroma-Vektordatenbank, die Texteinbettungen (Embeddings)
von Homers "Odyssee" enthält, und führt eine semantische Suche durch, um inhaltlich 
relevante Textabschnitte zu einer bestimmten Anfrage zu finden. Das Skript demonstriert,
wie semantische Ähnlichkeitssuche funktioniert, bei der Texte auch dann gefunden werden,
wenn sie keine exakten Schlüsselwörter, aber ähnliche Bedeutungen enthalten.

Voraussetzungen:
- Eine bestehende Chroma-Vektordatenbank im Verzeichnis "db/chroma_db"
- Zugang zu einem laufenden Ollama-Server
"""

import os  # Für den Zugriff auf Dateipfade und Umgebungsvariablen

# Import der notwendigen LangChain-Komponenten
from langchain_community.vectorstores import Chroma  # Vektordatenbank für die semantische Suche
from langchain_ollama import OllamaEmbeddings  # Tool zum Erstellen von Texteinbettungen

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells für konsistente Vektorisierung
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder Standard

# Definieren des Verzeichnisses, in dem die Vektordatenbank gespeichert ist
current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Verzeichnis des Skripts
persistent_directory = os.path.join(current_dir, "db", "chroma_db")  # Pfad zur gespeicherten Vektordatenbank

# Initialisieren des Embedding-Modells
# Es ist wichtig, dasselbe Embedding-Modell zu verwenden, mit dem die Datenbank erstellt wurde,
# da unterschiedliche Modelle verschiedene Vektorrepräsentationen erzeugen würden
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Laden der existierenden Vektordatenbank mit der Embedding-Funktion
# Die Datenbank enthält bereits die Textabschnitte und ihre Vektorrepräsentationen
# Die Embedding-Funktion wird benötigt, um die Suchanfrage in einen Vektor umzuwandeln
db = Chroma(
    persist_directory=persistent_directory,  # Ort der gespeicherten Datenbank
    embedding_function=embeddings  # Funktion zur Umwandlung von Text in Vektoren
)

# Definieren der Benutzeranfrage
# Diese Anfrage wird in einen Vektor umgewandelt und mit den gespeicherten Vektoren verglichen
query = "Who is Odysseus' wife?"  # Beispielanfrage zur Suche nach Odysseus' Ehefrau (Penelope)

# Abrufen relevanter Dokumente basierend auf der Anfrage
# Erstellen eines Retrievers mit spezifischen Suchparametern
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # Suche basierend auf Ähnlichkeitsschwellenwert
    search_kwargs={
        "k": 3,  # Maximal 3 Dokumente zurückgeben
        "score_threshold": 0.4,  # Nur Dokumente mit einer Ähnlichkeit > 0.4 zurückgeben
                                 # (0 = keine Ähnlichkeit, 1 = identisch)
    },
)
# Durchführung der eigentlichen Suche
# Die Anfrage wird in einen Vektor umgewandelt und mit den Vektoren in der Datenbank verglichen
relevant_docs = retriever.invoke(query)

# Anzeigen der relevanten Ergebnisse mit Metadaten
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    # Für jedes gefundene Dokument den Textinhalt anzeigen
    print(f"Document {i}:\n{doc.page_content}\n")
    # Falls Metadaten vorhanden sind, auch die Quelle anzeigen
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        # Die Metadaten enthalten in der Regel die Quelldatei und möglicherweise weitere Informationen