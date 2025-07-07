import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))  # Ermittelt das aktuelle Ausführungsverzeichnis des Skripts
db_dir = os.path.join(current_dir, "db")  # Hauptverzeichnis für die Datenbankablage

print("Creating vector store with custom Hugging Face embeddings...")
huggingface_custom_embeddings2 = HuggingFaceEmbeddings(
    model_name="oliverguhr/revosax-granite-embedding-278m-multilingual"  # Ein weiteres Modell für semantische Ähnlichkeit
)

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
query = "Wie starb Julia"  # Beispielfrage: "Wie starb Julia"

query_vector_store("chroma_db_huggingface_custom2", query, huggingface_custom_embeddings2)  # Zweite Custom Hugging Face-basierte Datenbank abfragen

print("Querying demonstrations completed.")  # Abschlussnachricht