import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever

# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
# Name des zu verwendenden Sprachmodells für die Texterzeugung
MODEL_NAME = "llama3.2"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")

# Embedding-Modell initialisieren
print("Creating vector store with custom Hugging Face embeddings...")
huggingface_custom_embeddings2 = HuggingFaceEmbeddings(
    model_name="oliverguhr/revosax-granite-embedding-278m-multilingual"
)

# Funktion zum Abfragen der Vektordatenbank
def query_vector_store(store_name, query, embedding_function):
    """
    Führt eine Abfrage auf der angegebenen Vektordatenbank durch und gibt die Ergebnisse aus.
    
    Args:
        store_name: Name der zu abfragenden Datenbank
        query: Die Anfrage als Text
        embedding_function: Das Embedding-Modell, das zur Vektorisierung der Anfrage verwendet werden soll
    """
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        # Chroma-Datenbank laden
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        # Standard-Retriever mit Ähnlichkeitssuche konfigurieren
        base_retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        # LLM für MultiQueryRetriever initialisieren
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)
        # MultiQueryRetriever initialisieren
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
        # Abfrage durchführen
        print(f"Query: {query}")
        relevant_docs = multi_query_retriever.invoke(query)
        # Ergebnisse anzeigen
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

# Definition der Benutzeranfrage
query = "Wie starb Julia"

# Vektordatenbank mit MultiQueryRetriever abfragen
query_vector_store("chroma_db_huggingface_custom2", query, huggingface_custom_embeddings2)

print("Querying demonstrations completed.")