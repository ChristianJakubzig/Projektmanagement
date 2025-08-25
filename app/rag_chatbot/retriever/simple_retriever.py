from typing import List, Dict, Any
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms.base import LLM

# Import fix
try:
    from ..ollama_client.client import OllamaClient
    from ..vector_store.chroma_client import get_chroma_vectorstore
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ollama_client.client import OllamaClient
    from vector_store.chroma_client import get_chroma_vectorstore

class OllamaLLM(LLM):
    """Wrapper damit Ollama mit LangChain funktioniert"""
    def __init__(self, ollama_client):
        super().__init__()
        self.client = ollama_client
    
    def _call(self, prompt: str, **kwargs) -> str:
        return self.client.chat(prompt, temperature=0.1)
    
    @property
    def _llm_type(self) -> str:
        return "ollama"

class SimpleRetriever:
    """Einfacher RAG mit MultiQuery"""
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        
        # Embedding-Funktion für Chroma
        class OllamaEmbeddings:
            def __init__(self, client):
                self.client = client
            def embed_documents(self, texts): 
                return [self.client.embed(text) for text in texts]
            def embed_query(self, text): 
                return self.client.embed(text)
        
        # Vectorstore aufbauen
        embeddings = OllamaEmbeddings(self.ollama_client)
        self.vectorstore = get_chroma_vectorstore(embeddings)
        
        # MultiQuery Retriever
        llm = OllamaLLM(self.ollama_client) 
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            llm=llm
        )
    
    def ask(self, question: str) -> str:
        """Stelle eine Frage und bekomme eine Antwort"""
        # 1. Dokumente finden
        docs = self.retriever.get_relevant_documents(question)
        
        # 2. Kontext bauen
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Antwort generieren
        prompt = f"""Kontext: {context}

Frage: {question}

Antwort:"""
        
        return self.ollama_client.chat(prompt)