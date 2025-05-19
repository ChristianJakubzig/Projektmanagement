"""
Dokumenten-Vektorisierungs-Tool für Textdateien

Dieses Skript dient zur Vektorisierung und Indizierung von Textdokumenten, um eine effiziente Suche zu ermöglichen.
Es lädt Textdateien aus einem angegebenen Verzeichnis, teilt sie in kleinere Chunks auf, erstellt Vektoreinbettungen 
mit dem Ollama-Embeddings-Modell und speichert diese in einer Chroma-Vektordatenbank. Falls die Datenbank bereits 
existiert, wird dieser Prozess übersprungen.

Funktionsweise:
1. Konfiguration von Pfaden und Modell-Einstellungen
2. Überprüfung, ob eine Vektordatenbank bereits existiert
3. Falls nicht: Laden der Textdateien, Aufteilen in Chunks, Erstellung der Vektoreinbettungen 
   und Speicherung in der Chroma-Datenbank
   
Die erstellte Vektordatenbank kann später für semantische Suchen verwendet werden.
"""

import os

# Import der benötigten Bibliothekenmodule
from langchain.text_splitter import CharacterTextSplitter  # Für das Aufteilen von Texten in kleinere Chunks
from langchain_community.document_loaders import TextLoader  # Zum Laden von Textdateien
from langchain_community.vectorstores import Chroma  # Vektordatenbank für die Speicherung der Embeddings
from langchain_ollama import OllamaEmbeddings  # Für die Erzeugung von Text-Embeddings mit Ollama

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert

# Definition der Verzeichnisse für Textdateien und die persistente Datenbank
current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Ausführungsverzeichnis des Skripts
books_dir = os.path.join(current_dir, "books")  # Verzeichnis, in dem die zu verarbeitenden Textdateien liegen
db_dir = os.path.join(current_dir, "db")  # Hauptverzeichnis für die Datenbank
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")  # Vollständiger Pfad zum Speicherort der Chroma-Datenbank

# Ausgabe der Verzeichnispfade zur Information
print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Überprüfung, ob die Chroma-Vektordatenbank bereits existiert
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")  # Meldung, dass die Datenbank neu erstellt werden muss

    # Sicherstellen, dass das Verzeichnis mit den Textdateien existiert
    if not os.path.exists(books_dir):
        # Fehler auslösen, wenn das Verzeichnis nicht gefunden wurde
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # Auflisten aller Textdateien im Verzeichnis (nur .txt-Dateien)
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Einlesen des Textinhalts aus jeder Datei und Speicherung mit Metadaten
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)  # Vollständiger Pfad zur Textdatei
        loader = TextLoader(file_path)  # Erstellen eines TextLoader-Objekts für die Datei
        book_docs = loader.load()  # Laden des Dokumentinhalts
        for doc in book_docs:
            # Hinzufügen von Metadaten zu jedem Dokument, um die Quelle zu kennzeichnen
            doc.metadata = {"source": book_file}
            documents.append(doc)  # Hinzufügen des Dokuments zur Liste

    # Aufteilen der Dokumente in kleinere Chunks für bessere Verarbeitung
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Erstellen eines Splitters mit Chunklänge 1000 Zeichen ohne Überlappung
    docs = text_splitter.split_documents(documents)  # Aufteilen der Dokumente in Chunks

    # Anzeigen von Informationen über die aufgeteilten Dokumente
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")  # Ausgabe der Anzahl der erstellten Chunks

    # Erstellen der Embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_URL
    )  # Initialisierung des Embedding-Modells mit den konfigurierten Einstellungen
    print("\n--- Finished creating embeddings ---")

    # Erstellen der Vektordatenbank und persistentes Speichern
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)  # Erstellung und Speicherung der Vektordatenbank
    print("\n--- Finished creating and persisting vector store ---")

else:
    # Falls die Datenbank bereits existiert, wird keine neue erstellt
    print("Vector store already exists. No need to initialize.")  # Meldung, dass keine neue Initialisierung nötig ist