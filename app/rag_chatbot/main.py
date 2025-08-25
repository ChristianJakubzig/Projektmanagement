"""
Hauptskript zum Testen des RAG-Systems
"""

# Einfacher Import-Fix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

from rag_chatbot.ollama_client.client import OllamaClient
from rag_chatbot.retriever.simple_retriever import SimpleRetriever
def test_ollama_connection():
    """Test 1: Ollama Chat"""
    print("=" * 50)
    print("🔗 Test 1: Ollama Chat")
    print("=" * 50)
   
    try:
        client = OllamaClient()
       
        # Verfügbare Modelle
        models = client.models()
        print(f"✅ Verfügbare Modelle: {len(models)}")
        for model in models:
            print(f"   - {model}")
       
        # Chat-Tests
        print("\n💬 Chat-Test:")
        response = client.chat("Hallo! Antworte in einem Satz.", temperature=0.7)
        print(f"Antwort: {response}")
        
        print("\n✅ Ollama Client funktioniert!")
        return True
       
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_rag_retriever():
    """Test 2: RAG mit MultiQuery"""
    print("\n" + "=" * 50)
    print("🧠 Test 2: RAG Retriever")
    print("=" * 50)
    
    try:
        print("🔄 Initialisiere RAG-System...")
        retriever = SimpleRetriever()
        
        print("✅ RAG-System bereit!")
        
        # Test-Fragen
        questions = [
            "Was ist Machine Learning?",
            "Erkläre künstliche Intelligenz",
            "Was sind neuronale Netze?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Frage {i}: {question}")
            print("🔍 Suche läuft...")
            
            answer = retriever.ask(question)
            print(f"🤖 Antwort: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG-System Fehler: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RAG-System Tests")
    
    # Test 1: Ollama
    ollama_ok = test_ollama_connection()
    
    if ollama_ok:
        # Test 2: RAG (nur wenn Ollama funktioniert)
        rag_ok = test_rag_retriever()
    else:
        print("\n⏭️  RAG-Tests übersprungen (Ollama-Fehler)")
    
    print("\n🏁 Tests abgeschlossen!")