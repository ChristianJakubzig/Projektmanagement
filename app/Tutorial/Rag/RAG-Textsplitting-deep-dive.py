"""
Vergleichstool für verschiedene Text-Splitter Methoden in Vektordatenbanken

Dieses Skript dient zum Vergleich verschiedener Textsplitting-Methoden für die Vektorisierung und Abfrage von Textdokumenten.
Es lädt eine Textdatei, teilt den Inhalt mit fünf verschiedenen Splitter-Methoden in Chunks auf, erstellt für jede Methode 
eine eigene Chroma-Vektordatenbank und führt anschließend die gleiche Abfrage auf allen Datenbanken durch, 
um die Unterschiede in den Ergebnissen zu vergleichen.

Funktionsweise:
1. Konfiguration von Pfaden und Modell-Einstellungen
2. Laden eines Textdokuments (Romeo und Julia)
3. Aufteilen des Dokuments mit verschiedenen Splitter-Methoden:
   - Character-based Splitting: Aufteilung nach fester Zeichenanzahl
   - Sentence-based Splitting: Aufteilung nach Satzgrenzen
   - Token-based Splitting: Aufteilung nach Tokens (Wörter/Subwörter)
   - Recursive Character-based Splitting: Intelligente Aufteilung nach natürlichen Grenzen
   - Custom Splitting: Benutzerdefinierte Aufteilung (hier nach Absätzen)
4. Erstellung separater Vektordatenbanken für jede Splitting-Methode
5. Durchführung einer identischen Abfrage auf allen Datenbanken
6. Vergleich der gefundenen Ergebnisse
"""

import os

# Import der benötigten Bibliothekenmodule
from langchain.text_splitter import (
    CharacterTextSplitter,  # Für die Aufteilung nach Zeichenanzahl
    RecursiveCharacterTextSplitter,  # Für die intelligente Aufteilung nach natürlichen Grenzen
    SentenceTransformersTokenTextSplitter,  # Für die Aufteilung nach Sätzen
    TextSplitter,  # Basisklasse für eigene Splitter
    TokenTextSplitter,  # Für die Aufteilung nach Tokens
)
from langchain_community.document_loaders import TextLoader  # Zum Laden von Textdateien
from langchain_community.vectorstores import Chroma  # Vektordatenbank für die Speicherung der Embeddings
from langchain_ollama import OllamaEmbeddings  # Für die Erzeugung von Text-Embeddings mit Ollama

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert

# Definition der Verzeichnisse und Dateipfade
current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Ausführungsverzeichnis des Skripts
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")  # Pfad zur Romeo und Julia Textdatei
db_dir = os.path.join(current_dir, "db")  # Hauptverzeichnis für die Datenbankablage

# Überprüfen, ob die Textdatei existiert
if not os.path.exists(file_path):
    # Fehler auslösen, wenn die Datei nicht gefunden wurde
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Einlesen des Textinhalts aus der Datei
loader = TextLoader(file_path)  # Erstellen eines TextLoader-Objekts für die Datei
documents = loader.load()  # Laden des Dokumentinhalts

# Definition des Embedding-Modells
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL, base_url=OLLAMA_URL
)  # Initialisierung des Embedding-Modells mit den konfigurierten Einstellungen


# Funktion zum Erstellen und Persistieren der Vektordatenbank
def create_vector_store(docs, store_name):
    """
    Erstellt eine Chroma-Vektordatenbank mit den übergebenen Dokumenten und speichert sie
    unter dem angegebenen Namen im Datenbankverzeichnis.
    
    Args:
        docs: Die zu speichernden Dokumente (bereits in Chunks aufgeteilt)
        store_name: Name für die zu erstellende Datenbank
    """
    persistent_directory = os.path.join(db_dir, store_name)  # Vollständiger Pfad zum Speicherort
    if not os.path.exists(persistent_directory):
        # Neue Datenbank erstellen, wenn sie noch nicht existiert
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        # Meldung ausgeben, wenn die Datenbank bereits existiert
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


# 1. Character-based Splitting (Zeichen-basierte Aufteilung)
# Teilt Text in Chunks basierend auf einer bestimmten Anzahl von Zeichen.
# Vorteil: Gleichmäßige Chunkgrößen unabhängig vom Inhalt
# Nachteil: Kann mitten in Sätzen oder Wörtern trennen, was die semantische Kohärenz beeinträchtigt
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # 1000 Zeichen pro Chunk mit 100 Zeichen Überlappung
char_docs = char_splitter.split_documents(documents)  # Dokumente in Chunks aufteilen
create_vector_store(char_docs, "chroma_db_char")  # Vektordatenbank erstellen

# 2. Sentence-based Splitting (Satz-basierte Aufteilung)
# Teilt Text in Chunks basierend auf Satzgrenzen, wodurch die semantische Einheit von Sätzen erhalten bleibt.
# Vorteil: Erhält die semantische Kohärenz innerhalb von Chunks, da an Satzgrenzen getrennt wird
# Nachteil: Unterschiedliche Chunkgrößen abhängig von der Satzlänge
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)  # Etwa 1000 Tokens pro Chunk
sent_docs = sent_splitter.split_documents(documents)  # Dokumente in Chunks aufteilen
create_vector_store(sent_docs, "chroma_db_sent")  # Vektordatenbank erstellen

# 3. Token-based Splitting (Token-basierte Aufteilung)
# Teilt Text in Chunks basierend auf Tokens (Wörtern oder Subwörtern), unter Verwendung von Tokenizern wie GPT-2.
# Vorteil: Optimiert für Transformer-Modelle mit strikten Token-Limits
# Nachteil: Möglicherweise weniger lesbare Chunks als bei satzbasierter Aufteilung
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)  # 512 Tokens pro Chunk ohne Überlappung
token_docs = token_splitter.split_documents(documents)  # Dokumente in Chunks aufteilen
create_vector_store(token_docs, "chroma_db_token")  # Vektordatenbank erstellen

# 4. Recursive Character-based Splitting (Rekursive zeichen-basierte Aufteilung)
# Versucht, Text an natürlichen Grenzen (Sätze, Absätze) innerhalb eines Zeichenlimits zu teilen.
# Vorteil: Balanciert zwischen Kohärenz und Einhaltung von Größengrenzwerten
# Nachteil: Komplexere Verarbeitung als einfache Aufteilungsmethoden
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)  # 1000 Zeichen pro Chunk mit 100 Zeichen Überlappung, aber Berücksichtigung der Textstruktur
rec_char_docs = rec_char_splitter.split_documents(documents)  # Dokumente in Chunks aufteilen
create_vector_store(rec_char_docs, "chroma_db_rec_char")  # Vektordatenbank erstellen

# 5. Custom Splitting (Benutzerdefinierte Aufteilung)
# Ermöglicht die Erstellung einer benutzerdefinierten Aufteilungslogik basierend auf spezifischen Anforderungen.
# Vorteil: Kann an spezielle Dokumentstrukturen angepasst werden, die mit Standardsplittern nicht optimal behandelt werden können
# Nachteil: Erfordert manuelle Implementierung und Anpassung
print("\n--- Using Custom Splitting ---")


class CustomTextSplitter(TextSplitter):
    """
    Benutzerdefinierter TextSplitter, der Text anhand von Absätzen aufteilt.
    
    Diese Klasse erweitert den TextSplitter und implementiert eine eigene
    split_text Methode, die den Text anhand von Leerzeilen (Absätzen) aufteilt.
    """
    def split_text(self, text):
        # Benutzerdefinierte Logik für die Textaufteilung
        return text.split("\n\n")  # Beispiel: Aufteilung nach Absätzen (leere Zeilen)


custom_splitter = CustomTextSplitter()  # Instanz des benutzerdefinierten Splitters erstellen
custom_docs = custom_splitter.split_documents(documents)  # Dokumente in Chunks aufteilen
create_vector_store(custom_docs, "chroma_db_custom")  # Vektordatenbank erstellen


# Funktion zum Abfragen einer Vektordatenbank
def query_vector_store(store_name, query):
    """
    Führt eine Abfrage auf der angegebenen Vektordatenbank durch und gibt die Ergebnisse aus.
    
    Args:
        store_name: Name der zu abfragenden Datenbank
        query: Die Anfrage als Text
    """
    persistent_directory = os.path.join(db_dir, store_name)  # Vollständiger Pfad zum Speicherort
    if os.path.exists(persistent_directory):
        # Datenbank abfragen, wenn sie existiert
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory, embedding_function=embeddings
        )  # Verbindung zur Chroma-Datenbank herstellen
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",  # Suche basierend auf Ähnlichkeitsschwellenwert
            search_kwargs={"k": 1, "score_threshold": 0.1},  # Parameter: maximal 1 Ergebnis mit mindestens 0.1 Ähnlichkeitswert
        )
        relevant_docs = retriever.invoke(query)  # Durchführung der Suche
        # Anzeigen der relevanten Ergebnisse mit Metadaten
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")  # Ausgabe des Dokumenteninhalts
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")  # Ausgabe der Quelle aus den Metadaten
    else:
        # Meldung ausgeben, wenn die Datenbank nicht existiert
        print(f"Vector store {store_name} does not exist.")


# Definition der Benutzeranfrage
# Diese Frage wird in einen Vektor umgewandelt und zum Suchen ähnlicher Dokumente verwendet
query = "How did Juliet die?"  # Beispielfrage: "Wie starb Julia?"

# Jede Vektordatenbank abfragen, um die Ergebnisse zu vergleichen
query_vector_store("chroma_db_char", query)  # Zeichen-basierte Datenbank abfragen
query_vector_store("chroma_db_sent", query)  # Satz-basierte Datenbank abfragen
query_vector_store("chroma_db_token", query)  # Token-basierte Datenbank abfragen
query_vector_store("chroma_db_rec_char", query)  # Rekursive zeichen-basierte Datenbank abfragen
query_vector_store("chroma_db_custom", query)  # Benutzerdefinierte Datenbank abfragen