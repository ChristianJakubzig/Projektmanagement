# RAG Chatbot 🤖

Ein Retrieval-Augmented Generation (RAG) Chatbot, der mit Ollama und ChromaDB arbeitet. Das System ermöglicht es, Fragen zu einer Sammlung von Dokumenten zu stellen und präzise Antworten basierend auf dem Dokumentinhalt zu erhalten.

## 📋 Übersicht

Dieses Projekt implementiert einen RAG-basierten Chatbot mit:
- **Streamlit** Web-Interface für einfache Bedienung
- **Ollama** für lokale LLM-Inferenz 
- **ChromaDB** als Vektordatenbank
- **LangChain** für RAG-Pipeline
- Vortrainierte Embeddings mit klassischer Literatur und historischen Dokumenten

## 🚀 Quick Start

### 1. Repository klonen
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Docker Container starten
Navigiere zum Projektordner und starte Docker Compose:
```bash
cd Projektmanagement
docker compose up -d
```
⏳ *Diese Aktion kann einige Minuten dauern...*

### 3. Development Container öffnen
1. Öffne VS Code
2. Drücke `F1`
3. Gib ein: `Dev Containers: Open Folder in Container`
4. Wähle den Projektordner aus

⏳ *Das Setup des Containers kann ebenfalls einige Minuten dauern...*

### 4. RAG Bot starten
Navigiere in der Container-Console zum Bot-Verzeichnis:
```bash
cd für_mich
streamlit run main.py
```

Die Anwendung öffnet sich automatisch im Browser unter `http://localhost:8501`

### 5. Bot stoppen
Um den Streamlit-Server zu stoppen:
```bash
Strg + C
```

### 6. Container verlassen
Um aus dem Development Container zurück zur lokalen Umgebung zu wechseln:
1. Klicke in VS Code **unten links** auf die **blaue Fläche** (Container-Indikator)
2. Wähle im Command Palette: `Reopen in SSH` aus

Alternativ:
- Drücke `F1` → `Dev Containers: Reopen Locally`

## 📁 Projektstruktur

```
.
├── data/                          # Datenverzeichnis
│   ├── raw_data_books/           # Literatur und historische Texte
│   │   ├── adventures_of_huckleberry_finn.txt
│   │   ├── romeo_and_juliet.txt
│   │   ├── declaration_of_independence_of_the_united_states.txt
│   │   └── ... (weitere klassische Texte)
│   └── raw_json_data/            # JSON-Datenfiles
│
├── für_mich/                     # 🎯 HAUPTVERZEICHNIS
│   ├── main.py                   # Streamlit App (HIER STARTEN!)
│   ├── chroma_client.py          # ChromaDB Verbindung
│   ├── config.py                 # Konfiguration
│   ├── embedding.py              # Embedding-Funktionen  
│   └── ...
│
├── rag_chatbot/                  # Erweiterte RAG-Implementierung
├── logs/                         # Log-Files
└── Projektmanagement/            # Docker Setup
```

## 🔧 Features

### 🎨 Streamlit Interface
- **Chat-basierte Benutzeroberfläche**
- **Quellenanzeige** für Transparenz
- **Chat-Verlauf** mit Löschfunktion

### 🧠 RAG-Pipeline
- **Ollama Integration** für lokale LLM-Inferenz
- **ChromaDB** als Vektordatenbank
- **LangChain** für nahtlose RAG-Implementation
- **Granite Embeddings** (278m Parameter)
- **Llama 3.2** als Chat-Model

### 📚 Vortrainierte Daten
Das System kommt mit einer Sammlung klassischer Literatur:
- Shakespeare (Romeo & Julia)
- Mark Twain (Huckleberry Finn, Tom Sawyer)
- Charles Dickens (Tale of Two Cities)
- Mary Shelley (Frankenstein)
- Herman Melville (Moby Dick)
- Jane Austen (Pride & Prejudice)
- Homer (Iliad, Odyssey)
- Leo Tolstoy (War & Peace)
- Historische Dokumente (US Declaration of Independence, Bill of Rights)

## ⚙️ Konfiguration

### Retriever-Einstellungen
- **k**: Anzahl der abgerufenen Dokumente (1-10)
- **Score Threshold**: Minimale Ähnlichkeit für Dokumente (0.0-1.0)

## 🛠️ Entwicklung

### Wichtige Dateien
- `main.py`: Streamlit Hauptanwendung
- `chroma_client.py`: Vektordatenbank-Verbindung
- `config.py`: Systemkonfiguration
- `embedding.py`: Document Processing & Embeddings

## 🔍 Verwendung

1. **Starte die App** wie oben beschrieben
2. **Stelle Fragen** über das Chat-Interface, z.B.:
   - "Wie stirbt Romeo in Romeo und Julia?"
   - "Was besagt die amerikanische Unabhängigkeitserklärung?"
   - "Beschreibe den Charakter von Huckleberry Finn"
3. **Experimentiere** mit verschiedenen System-Prompts und Retriever-Einstellungen
4. **Überprüfe Quellen** über die ausklappbaren Bereiche

## 📝 Hinweise

- **Erste Anfrage**: Kann länger dauern, da Modelle geladen werden
- **Docker Resources**: Stelle sicher, dass Docker genügend RAM zugewiesen hat
- **Ollama Server**: Läuft auf einem externen Server (TH Wildau)
- **Datenbank**: ChromaDB wird lokal im Container gespeichert

## 🆘 Troubleshooting

**Streamlit App lädt nicht?**  
- Warte nach Container-Start 2-3 Minuten
- Überprüfe die Console-Ausgabe

**Keine Antworten vom Bot?**
- Überprüfe Internetverbindung (für Ollama Server)
- Erhöhe den Score Threshold in den Einstellungen

**Leere Antworten?**
- Senke den Score Threshold
- Verwende andere Retriever-Modi
- Formuliere Fragen spezifischer