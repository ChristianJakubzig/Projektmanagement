"""
Vergleichstool für verschiedene Embedding-Modelle in Vektordatenbanken

Dieses Skript dient zum Vergleich verschiedener Embedding-Modelle für die Vektorisierung und Abfrage von Textdokumenten.
Es lädt eine Textdatei (Odyssee), teilt den Inhalt in Chunks auf und erstellt dann zwei separate Vektordatenbanken
unter Verwendung unterschiedlicher Embedding-Modelle: Ollama (lokale KI) und Hugging Face (vortrainierte Transformermodelle).
Anschließend wird die gleiche Abfrage auf beiden Datenbanken durchgeführt, um die Qualität der Suchergebnisse zu vergleichen.

Funktionsweise:
1. Konfiguration von Pfaden und Modell-Einstellungen
2. Laden eines Textdokuments (Odyssee)
3. Aufteilen des Dokuments in Chunks von 1000 Zeichen
4. Erstellung zweier Vektordatenbanken mit unterschiedlichen Embedding-Modellen:
   - Ollama (lokales Modell): nomic-embed-text
   - Hugging Face (Cloud/lokal): sentence-transformers/all-mpnet-base-v2
5. Durchführung einer identischen Abfrage auf beiden Datenbanken
6. Vergleich der gefundenen Ergebnisse
"""

import os

# Import der benötigten Bibliothekenmodule
from langchain.embeddings import HuggingFaceEmbeddings  # Für Hugging Face basierte Embeddings
from langchain.text_splitter import CharacterTextSplitter  # Für die Aufteilung nach Zeichenanzahl
from langchain_community.document_loaders import TextLoader  # Zum Laden von Textdateien
from langchain_community.vectorstores import Chroma  # Vektordatenbank für die Speicherung der Embeddings
from langchain_ollama import OllamaEmbeddings  # Für die Erzeugung von Text-Embeddings mit Ollama

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert

# Definition der Verzeichnisse für die Textdatei und die persistente Datenbank
current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Ausführungsverzeichnis des Skripts
file_path = os.path.join(current_dir, "books", "odyssey.txt")  # Pfad zur Odyssee Textdatei
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

# Aufteilen des Dokuments in Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # 1000 Zeichen pro Chunk ohne Überlappung
docs = text_splitter.split_documents(documents)  # Dokumente in Chunks aufteilen

# Anzeigen von Informationen über die aufgeteilten Dokumente
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")  # Anzahl der erstellten Chunks ausgeben
print(f"Sample chunk:\n{docs[0].page_content}\n")  # Beispiel eines Chunks anzeigen


# Funktion zum Erstellen und Persistieren der Vektordatenbank
def create_vector_store(docs, embeddings, store_name):
    """
    Erstellt eine Chroma-Vektordatenbank mit den übergebenen Dokumenten und dem angegebenen Embedding-Modell
    und speichert sie unter dem angegebenen Namen im Datenbankverzeichnis.
    
    Args:
        docs: Die zu speichernden Dokumente (bereits in Chunks aufgeteilt)
        embeddings: Das zu verwendende Embedding-Modell
        store_name: Name für die zu erstellende Datenbank
    """
    persistent_directory = os.path.join(db_dir, store_name)  # Vollständiger Pfad zum Speicherort
    if not os.path.exists(persistent_directory):
        # Neue Datenbank erstellen, wenn sie noch nicht existiert
        print(f"\n--- Creating vector store {store_name} ---")
        # Hier findet der eigentliche Embedding-Prozess statt:
        # Jeder Textchunk wird durch das übergebene Embedding-Modell in einen numerischen Vektor umgewandelt
        # und anschließend in der Chroma-Datenbank gespeichert
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        # Meldung ausgeben, wenn die Datenbank bereits existiert
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


# 1. Ollama Embeddings
# Verwendet lokale Embedding-Modelle, die über Ollama bereitgestellt werden.
# Vorteil: Kostenlos, lokal ausführbar ohne externe API, volle Datenkontrolle
# Nachteil: Möglicherweise weniger akkurat als Cloud-basierte Modelle, benötigt lokale Rechenleistung
print("\n--- Using Ollama Embeddings ---")
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)  # Initialisierung des Ollama Embedding-Modells
create_vector_store(docs, ollama_embeddings, "chroma_db_ollama")  # Vektordatenbank mit Ollama Embeddings erstellen

# 2. Hugging Face Transformers
# Verwendet Modelle aus der Hugging Face Bibliothek.
# Vorteil: Große Auswahl an hochqualitativen Modellen für verschiedene Aufgaben und Sprachen
# Nachteil: Größere Modelle benötigen mehr Arbeitsspeicher und Rechenleistung
# Hinweis: Weitere Modelle unter https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Ein leistungsstarkes Sentence-Transformer Modell für semantische Ähnlichkeit
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")  # Vektordatenbank mit Hugging Face Embeddings erstellen

print("Embedding demonstrations for Ollama and Hugging Face completed.")


# Funktion zum Abfragen einer Vektordatenbank
def query_vector_store(store_name, query, embedding_function):
    """
    Führt eine Abfrage auf der angegebenen Vektordatenbank durch und gibt die Ergebnisse aus.
    
    Args:
        store_name: Name der zu abfragenden Datenbank
        query: Die Anfrage als Text
        embedding_function: Das Embedding-Modell, das zur Vektorisierung der Anfrage verwendet werden soll
                           (muss dasselbe sein wie bei der Erstellung der Datenbank)
    """
    persistent_directory = os.path.join(db_dir, store_name)  # Vollständiger Pfad zum Speicherort
    if os.path.exists(persistent_directory):
        # Datenbank abfragen, wenn sie existiert
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,  # Wichtig: Dasselbe Embedding-Modell wie bei der Erstellung verwenden
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",  # Suche basierend auf Ähnlichkeitsschwellenwert
            search_kwargs={"k": 3, "score_threshold": 0.1},  # Parameter: maximal 3 Ergebnisse mit mindestens 0.1 Ähnlichkeitswert
        )
        # Bei der Abfrage wird der Anfragetext ebenfalls durch das Embedding-Modell in einen Vektor umgewandelt,
        # und dann werden die ähnlichsten Dokumente aus der Datenbank zurückgegeben
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
query = "Who is Odysseus' wife?"  # Beispielfrage: "Wer ist Odysseus' Frau?"

# Jede Vektordatenbank abfragen, um die Ergebnisse zu vergleichen
# Wichtig: Für jede Datenbank wird das entsprechende Embedding-Modell verwendet,
# mit dem die Datenbank erstellt wurde
query_vector_store("chroma_db_ollama", query, ollama_embeddings)  # Ollama-basierte Datenbank abfragen
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)  # Hugging Face-basierte Datenbank abfragen

print("Querying demonstrations completed.")  # Abschlussnachricht