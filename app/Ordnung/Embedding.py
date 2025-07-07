import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Ausführungsverzeichnis des Skripts
file_path = os.path.join(current_dir, "books", "Romeo&Julia.txt")  # Pfad zur Romeo&Julia Textdatei
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
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # 1000 Zeichen pro Chunk mit 200 Zeichen Überlappung
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
        Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persistent_directory,
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        # Meldung ausgeben, wenn die Datenbank bereits existiert
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

print("\n--- Using Custom Hugging Face Transformers Model ---")
huggingface_custom_embeddings = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-278m-multilingual"  # Ein weiteres leistungsstarkes Modell für semantische Ähnlichkeit
)

# 4. Erstellen der Vektordatenbank mit benutzerdefinierten Hugging Face Embeddings
print("Creating vector store with custom Hugging Face embeddings...")
huggingface_custom_embeddings2 = HuggingFaceEmbeddings(
    model_name="oliverguhr/revosax-granite-embedding-278m-multilingual"  # Ein weiteres Modell für semantische Ähnlichkeit
)

create_vector_store(docs, huggingface_custom_embeddings2, "chroma_db_huggingface_custom2")  # Vektordatenbank mit benutzerdefinierten Hugging Face Embeddings erstellen