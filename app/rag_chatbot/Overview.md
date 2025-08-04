## 📁 Projektstruktur

```plaintext
/app/
├── rag_chatbot/
│   ├── __init__.py
│   ├── config.py                 # Konfiguration und Umgebungsvariablen
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedding_manager.py  # Embedding-Generierung
│   │   └── document_processor.py # Dokumentenverarbeitung
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── chroma_client.py      # ChromaDB Verbindung
│   │   └── vector_operations.py  # Such- und Speicheroperationen
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py          # RAG Retrieval Logic
│   │   └── query_processor.py    # Query-Optimierung
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_client.py         # LLM Integration (llama)
│   │   └── prompt_templates.py   # Prompt Management
│   ├── chat/
│   │   ├── __init__.py
│   │   ├── chatbot.py            # Hauptchatbot-Klasse
│   │   └── conversation.py       # Conversation Management
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py     # Logging Setup
│       └── helpers.py            # Hilfsfunktionen
├── data/                         # Dokumente für Indexierung
├── tests/                        # Test-Suite
└── main.py                       # Hauptanwendung / CLI / Streamlit-Einstiegspunkt
