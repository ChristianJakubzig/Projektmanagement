# Import der benötigten Bibliotheken
import os  # Bietet Funktionen zur Interaktion mit dem Betriebssystem (Dateipfade, Verzeichnisse)
from langchain_ollama import ChatOllama  # Ermöglicht die Verwendung von Ollama-basierten Chat-Modellen
from langchain.text_splitter import CharacterTextSplitter  # Tool zum Aufteilen von Texten in kleinere Stücke
from langchain_community.document_loaders import TextLoader  # Lädt Textdateien als Dokumente
from langchain_community.vectorstores import Chroma  # Vektorbasierte Datenbank für Texteinbettungen
from langchain_ollama import OllamaEmbeddings  # Erstellt Texteinbettungen mit Ollama-Modellen

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert

# Ermittlung der Dateipfade
# Bestimmt das aktuelle Verzeichnis der ausgeführten Datei
current_dir = os.path.dirname(os.path.abspath(__file__))
# Definiert den Pfad zur Textdatei (Homers Odyssee)
file_path = os.path.join(current_dir, "books", "odyssey.txt")
# Definiert den Pfad zum Speichern der Vektordatenbank
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Überprüfung, ob die Chroma-Vektordatenbank bereits existiert
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    # Das Verzeichnis für die Vektordatenbank existiert nicht, also müssen wir sie erstellen

    # Sicherstellung, dass die Textdatei existiert
    if not os.path.exists(file_path):
        # Wenn die Datei nicht gefunden wird, wird eine Ausnahme ausgelöst
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Laden des Textinhalts aus der Datei
    # TextLoader wandelt die Textdatei in ein Dokument-Objekt um, das von LangChain verarbeitet werden kann
    loader = TextLoader(file_path)
    documents = loader.load()  # Lädt den gesamten Text als Dokument
    # In Chroma (und ähnlichen Vektordatenbanken) wird jeder Eintrag als Tripel gespeichert:
    #   ID: Ein eindeutiger Identifikator für jeden Textabschnitt
    #   Vektor: Das Embedding des Textabschnitts
    #   Metadaten + Text: Der eigentliche Textinhalt und zusätzliche Informationen
    # Im vorgestellten Code werden die Standardmetadaten verwendet, die vom TextLoader und dem Splitting-Prozess erzeugt werden, darunter:
    #   Quelldateipfad
    #   Positionsangaben im Text

    # Aufteilung des Dokuments in kleinere Chunks (Textabschnitte)
    # Dies ist wichtig, da große Textdokumente nicht am Stück verarbeitet werden können
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # chunk_size: Maximale Anzahl von Zeichen pro Chunk
    # chunk_overlap: Keine Überlappung zwischen den Chunks (0 Zeichen)
    docs = text_splitter.split_documents(documents)  # Teilt das Dokument in kleine Abschnitte

    # Anzeige von Informationen über die geteilten Dokumente
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")  # Zeigt die Anzahl der erstellten Chunks an
    print(f"Sample chunk:\n{docs[0].page_content}\n")  # Zeigt den Inhalt des ersten Chunks als Beispiel

    # Erstellung der Einbettungen (Embeddings)
    # Einbettungen sind numerische Vektordarstellungen von Text, die semantische Ähnlichkeiten erfassen
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL, base_url=OLLAMA_URL
    )  # Konfiguration des Embedding-Modells mit dem definierten Modellnamen und der Server-URL
    print("\n--- Finished creating embeddings ---")

    # Erstellung der Vektordatenbank und automatisches Speichern
    # Die Vektordatenbank ermöglicht semantische Suche und Ähnlichkeitsabfragen
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )  # Erstellt eine Chroma-Datenbank aus den Dokumenten und Einbettungen und speichert sie im angegebenen Verzeichnis
    print("\n--- Finished creating vector store ---")

else:
    # Die Vektordatenbank existiert bereits, keine Initialisierung erforderlich
    print("Vector store already exists. No need to initialize.")
    # In diesem Fall wird die Vektordatenbank nicht neu erstellt, um Rechenzeit zu sparen
    # Sie kann später geladen und verwendet werden