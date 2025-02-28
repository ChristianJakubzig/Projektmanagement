#import nltk
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')

import os
import logging
import json
from ollama import Client

# Setze die Ollama-URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
ollama = Client(host=OLLAMA_URL)

# Konfiguriere das Logging
logging.basicConfig(level=logging.INFO)

def test_ollama():
    """Testet, ob das LLM eine valide JSON-Antwort gibt."""
    question = "Gib die Hauptstadt von Deutschland als JSON zurück. Beispiel: {\"capital\": \"Berlin\"}"

    try:
        # Anfrage an das LLM mit JSON-Format
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": question}],
            format="json"  # JSON-Format explizit anfordern
        )

        # Logge die gesamte Antwort
        logging.info("Antwort vom LLM: %s", response)

        # Überprüfe, ob die Antwort wirklich JSON ist
        try:
            parsed_response = json.loads(response.message.content)
            print("Valides JSON erhalten:", parsed_response)
        except json.JSONDecodeError:
            print("Fehler: Antwort ist kein valides JSON!")

    except Exception as e:
        logging.error(f"Fehler bei der Kommunikation mit Ollama: {e}")

if __name__ == "__main__":
    test_ollama()


