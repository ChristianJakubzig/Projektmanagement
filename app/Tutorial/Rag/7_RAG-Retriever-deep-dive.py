"""
Vektorsuche in einer Chroma-Datenbank

Dieses Skript demonstriert verschiedene Suchmethoden in einer Vektordatenbank
unter Verwendung von Chroma und Ollama-Embeddings. Es lädt eine bestehende
Vektordatenbank und führt verschiedene Arten von Abfragen durch, um die
Unterschiede zwischen den Suchmethoden zu veranschaulichen.

Die unterstützten Suchmethoden sind:
1. Ähnlichkeitssuche (Similarity Search)
2. Max Marginal Relevance (MMR)
3. Ähnlichkeitsschwellenwert (Similarity Score Threshold)

Die Ergebnisse jeder Suchmethode werden mit den zugehörigen Metadaten angezeigt.
"""

import os

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Konfigurationsvariablen
# Festlegung des zu verwendenden Embedding-Modells für die Umwandlung von Text in Vektoren
EMBEDDING_MODEL = "nomic-embed-text"  # Dieses Modell wandelt Text in numerische Vektoren um
# URL des Ollama-Servers, aus Umgebungsvariable oder Standardwert
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Verwendet die OLLAMA_URL Umgebungsvariable oder den Standardwert, wenn nicht gesetzt

# Definition des Verzeichnisses für die persistente Speicherung der Vektordatenbank
# Bestimmt zuerst das aktuelle Verzeichnis des Skripts
current_dir = os.path.dirname(os.path.abspath(__file__))
# Erstellt einen Pfad zum Datenbankverzeichnis
db_dir = os.path.join(current_dir, "db")
# Vollständiger Pfad zum Chroma-Datenbankverzeichnis mit Metadaten
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Initialisierung des Embedding-Modells
# Erstellt ein OllamaEmbeddings-Objekt mit dem konfigurierten Modell und der Server-URL
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Laden der bestehenden Vektordatenbank mit der definierten Embedding-Funktion
# Chroma lädt die Datenbank aus dem angegebenen Verzeichnis und verwendet die Embedding-Funktion
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


# Funktion zur Abfrage einer Vektordatenbank mit verschiedenen Suchtypen und Parametern
def query_vector_store(
    store_name,  # Name des Vektorspeichers für Ausgabezwecke
    query,       # Die Suchanfrage als Text
    embedding_function,  # Die Funktion zur Umwandlung von Text in Vektoren
    search_type,  # Der zu verwendende Suchtyp (similarity, mmr, similarity_score_threshold)
    search_kwargs  # Zusätzliche Parameter für die Suche (z.B. k, fetch_k, lambda_mult, score_threshold)
):
    # Prüft, ob das angegebene Verzeichnis existiert
    if os.path.exists(persistent_directory):
        print(f"\n--- Abfrage des Vektorspeichers {store_name} ---")
        
        # Lädt die Chroma-Vektordatenbank aus dem persistenten Verzeichnis
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        
        # Erstellt einen Retriever mit dem angegebenen Suchtyp und den Suchparametern
        retriever = db.as_retriever(
            search_type=search_type,  # Typ der Vektorsuche
            search_kwargs=search_kwargs,  # Zusätzliche Parameter für die Suche
        )
        
        # Führt die Suchanfrage durch und erhält relevante Dokumente
        relevant_docs = retriever.invoke(query)
        
        # Zeigt die relevanten Ergebnisse mit Metadaten an
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


# Definition der Benutzeranfrage
query = "How did Juliet die?"  # Beispielanfrage: "Wie starb Julia?"

# Demonstration verschiedener Abrufmethoden

# 1. Ähnlichkeitssuche (Similarity Search)
# Diese Methode ruft Dokumente basierend auf der Vektorähnlichkeit ab.
# Sie findet die ähnlichsten Dokumente zum Abfragevektor basierend auf der Kosinus-Ähnlichkeit.
# Verwenden Sie diese Methode, wenn Sie die k ähnlichsten Dokumente abrufen möchten.
print("\n--- Verwendung der Ähnlichkeitssuche ---")
query_vector_store("chroma_db_with_metadata", query,
                   embeddings, "similarity", {"k": 3})  # Ruft die 3 ähnlichsten Dokumente ab

# 2. Max Marginal Relevance (MMR)
# Diese Methode balanciert zwischen der Auswahl von Dokumenten, die für die Anfrage relevant sind,
# und der Diversität unter den ausgewählten Dokumenten.
# 'fetch_k' gibt die Anzahl der initial abzurufenden Dokumente basierend auf Ähnlichkeit an.
# 'lambda_mult' steuert die Diversität der Ergebnisse: 1 für minimale Diversität, 0 für maximale Diversität.
# Verwenden Sie diese Methode, um Redundanz zu vermeiden und vielfältige, aber relevante Dokumente abzurufen.
# Hinweis: Relevanz misst, wie genau Dokumente zur Anfrage passen.
# Hinweis: Diversität stellt sicher, dass die abgerufenen Dokumente nicht zu ähnlich zueinander sind,
#          und bietet somit eine breitere Palette an Informationen.
print("\n--- Verwendung von Max Marginal Relevance (MMR) ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "mmr",
    {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},  # Holt 3 aus initial 20 Dokumenten mit Diversitätsgewichtung 0.5
)

# 3. Ähnlichkeitsschwellenwert (Similarity Score Threshold)
# Diese Methode ruft Dokumente ab, die einen bestimmten Ähnlichkeitsschwellenwert überschreiten.
# 'score_threshold' legt den Mindestwert für die Ähnlichkeit fest, den ein Dokument haben muss, 
# um als relevant betrachtet zu werden.
# Verwenden Sie diese Methode, wenn Sie sicherstellen möchten, dass nur hochrelevante Dokumente 
# abgerufen werden, während weniger relevante herausgefiltert werden.
print("\n--- Verwendung des Ähnlichkeitsschwellenwerts ---")
query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "similarity_score_threshold",
    {"k": 3, "score_threshold": 0.1},  # Holt bis zu 3 Dokumente mit Ähnlichkeitswert > 0.1
)

print("Demonstration der Abfragen mit verschiedenen Suchtypen abgeschlossen.")