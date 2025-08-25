import requests
from typing import List, Optional

# Import fix - funktioniert sowohl als Modul als auch direkt
try:
    # Erst versuchen als relativer Import
    from ..config import Config
except ImportError:
    # Dann als absoluter Import
    try:
        from config import Config  
    except ImportError:
        # Fallback: sys.path erweitern
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from config import Config

class OllamaClient:
    """Einfacher Ollama Client für LLM und Embedding Anfragen"""
    
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL.rstrip('/')
        
    def chat(self, message: str, model: Optional[str] = None, temperature: float = 0.7, top_p: float = 0.9, **kwargs) -> str:
        """Einfacher Chat - gibt nur die Antwort zurück"""
        url = f"{self.base_url}/api/chat"
        data = {
            "model": model or Config.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": message}],
            "stream": False,  # Wichtig: False für einfache Antwort
            "keep_alive": Config.OLLAMA_KEEP_ALIVE,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json()["message"]["content"]
    
    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Erstellt Embedding für Text"""
        url = f"{self.base_url}/api/embeddings"
        data = {
            "model": model or Config.OLLAMA_EMBEDDING_MODEL,
            "prompt": text,
            "keep_alive": Config.OLLAMA_KEEP_ALIVE
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json()["embedding"]
    
    def models(self) -> List[str]:
        """Liste verfügbare Modelle"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        
        return [model["name"] for model in response.json()["models"]]