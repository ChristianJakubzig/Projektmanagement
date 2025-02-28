from ollama import Client

OLLAMA_URL = "http://ollama:11434"
ollama = Client(host=OLLAMA_URL)

EMBEDDING_MODEL = "nomic-embed-text"
MODEL_NAME = "llama3.2"

try:
    # Versuche, das Modell zu ziehen
    ollama.pull(EMBEDDING_MODEL)
    print(f"✅ Modell {EMBEDDING_MODEL} erfolgreich gezogen!")
    ollama.pull(MODEL_NAME)
    print(f"✅ Modell {MODEL_NAME} erfolgreich gezogen!")
except Exception as e:
    print(f"❌ Fehler beim Ziehen des Modells: {e}")

