from langchain_ollama import OllamaEmbeddings
from chroma_client import get_chroma_vectorstore
from langchain_ollama import ChatOllama
from config import Config

emb = OllamaEmbeddings(
    base_url="https://ollama-bim24.apps.rhos.th-wildau.de",
    model="granite-embedding:278m",
    keep_alive=300,
    )

llm = ChatOllama(
    base_url="https://ollama-bim24.apps.rhos.th-wildau.de",
    model="llama3.2",
    keep_alive="5m",
    temperature=0.7,
)

db = get_chroma_vectorstore(emb)

def query_vectorstore(
        store_name,
        query,
        embedding_model,
        search_type,
        search_kwargs
):
    if store_name == Config.CHROMA_COLLECTION_NAME:
        print(f"\n--- Abfrage des Vektorspeichers {store_name} ---")
        db = get_chroma_vectorstore(embedding_model)

        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        relevant_docs = retriever.invoke(query)

        print(f"\n--- Relevante Dokumente für {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            # Zeigt den Inhalt des Dokuments an
            print(f"Dokument {i}:\n{doc.page_content}\n")
            # Zeigt die Quelle aus den Metadaten an, falls vorhanden
            if doc.metadata:
                print(f"Quelle: {doc.metadata.get('source', 'Unbekannt')}\n")
    else:
        # Fehlermeldung, wenn der Vektorspeicher nicht existiert
        print(f"Vektorspeicher {store_name} existiert nicht.")


query = "Tell me how does the first character is called that cpt. Ahab meets in moby dick?"

print("\n--- Verwendung des Ähnlichkeitsschwellenwerts ---")
query_vectorstore(
    Config.CHROMA_COLLECTION_NAME,
    query,
    emb,
    "similarity_score_threshold",
    {"k": 3, "score_threshold": 0.7},  # Holt bis zu 3 Dokumente mit Ähnlichkeitswert > 0.1
)

# Debug: Zeig mal kurz die Quellen in deiner Collection
col = db._collection  # Chroma-Collection (aus LangChain-Wrapper)
print("Docs gesamt:", col.count())
sample = col.get(include=["metadatas"], limit=10)
sources = { (m or {}).get("source", "unbekannt") for m in sample.get("metadatas", []) }
print("Beispiel-Quellen:", sources)


# Alle Dokumente mit Metadaten abrufen
results = col.get(
    include=["metadatas", "documents"]
)

# Datenquellen aus Metadaten extrahieren
sources = set()
for metadata in results['metadatas']:
    if metadata and 'source' in metadata:
        sources.add(metadata['source'])

print("Gefundene Datenquellen:")
for source in sorted(sources):
    print(f"- {source}")

col = db._collection
print(f"\n=== COLLECTION INFO ===")
print(f"Docs gesamt: {col.count()}")
print(f"Collection Name: {col.name}")