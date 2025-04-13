# Dokumentation: RAG-Bot Projekt mit Tkinter GUI

## Überblick
Dieses Projekt ist eine Retrieval-Augmented Generation (RAG) Anwendung, die Dokumentenverarbeitung, eine Vektordatenbank und ein lokales Sprachmodell kombiniert, um kontextbezogene Antworten auf Benutzeranfragen zu liefern. Es verwendet eine Microservices-Architektur mit Docker Compose und integriert Dienste wie PostgreSQL, Ollama, ChromaDB und eine benutzerdefinierte Python-Anwendung. Die Benutzeroberfläche wird über eine Tkinter-basierte GUI bereitgestellt.

### Projektziele
- Verarbeitung und Indizierung von PDF-Dokumenten (z. B. `BOI.pdf`).
- Bereitstellung einer API für Chat-Interaktionen mit serverseitig verwaltetem Verlauf.
- Nutzung eines lokalen Sprachmodells (LLM) für Antwortgenerierung.
- Integration einer Vektordatenbank für Retrieval-Augmented Generation.
- Einfache Desktop-Benutzeroberfläche mit Tkinter.

### Verzeichnisstruktur
```
C:.
├───.devcontainer           # Konfiguration für Entwicklungsumgebung (VS Code Dev Container)
├───app                     # Hauptverzeichnis der Python-Anwendung
│   ├───data                # Speicherort für Eingabedateien (z. B. BOI.pdf)
│   ├───logs                # Log-Dateien der Anwendung
│   ├───vectorstore         # Persistenter Speicher der Vektordatenbank
│   │   ├───0805637c-...    # UUID-Ordner für ChromaDB-Daten
│   │   └───6c31c2dd-...    # UUID-Ordner für ChromaDB-Daten
│   └───__pycache__         # Python-Cache-Dateien
├───data                    # Zusätzlicher Datenordner (außerhalb von app/)
└───logs                    # Zusätzlicher Log-Ordner (außerhalb von app/)
```

## Architektur

### Komponenten
1. **PostgreSQL** (`pgdatabase`):
   - Datenbank für persistente Speicherung (z. B. Metadaten oder Konfigurationen).
   - Konfiguriert mit Benutzer `myuser`, Passwort `mypassword` und Datenbank `mydatabase`.

2. **Ollama** (`ollama`):
   - Lokales Sprachmodell (LLM), standardmäßig `llama3.2`.
   - Bereitgestellt über Port `11434`.

3. **ChromaDB** (`chromadb_container`):
   - Vektordatenbank zur Speicherung und Abfrage von Dokumenten-Embeddings.
   - Standardport `8000`.

4. **Python-Anwendung** (`python_app_container`):
   - Hauptlogik der RAG-Anwendung, implementiert mit FastAPI.
   - Verarbeitet PDF-Dokumente, erstellt Embeddings, verwaltet den Chat-Verlauf und interagiert mit Ollama und ChromaDB.
   - API-Endpunkt: `/api/chat`.

5. **Tkinter GUI** (`chatbot_gui.py`):
   - Einfache Desktop-Benutzeroberfläche zur Interaktion mit der API.
   - Sendet Anfragen an `http://localhost:8000/api/chat` und zeigt Antworten in einem scrollbaren Textfeld an.

### Netzwerk
Alle Dienste (außer der Tkinter GUI) sind über ein benutzerdefiniertes Docker-Netzwerk `umgebungschaffen_net` verbunden, das die Kommunikation zwischen Containern ermöglicht. Die Tkinter GUI läuft lokal auf dem Host und kommuniziert über HTTP mit der API.

## Technische Details

### Docker Compose (`docker-compose.yml`)
Definiert die Dienste, ihre Abhängigkeiten, Ports, Umgebungsvariablen und Volumes:
- **Postgres**: Persistente Daten in `pgdata`.
- **Ollama**: LLM mit langer Keep-Alive-Zeit (`24h`).
- **ChromaDB**: Speichert Vektordaten in `chromadb_data`.
- **Python App**: Abhängig von Postgres (gesund), Ollama und ChromaDB (gestartet).

### Dockerfile (`Dockerfile`)
- Basis: `python:3.11-slim`.
- Installiert Python-, C++- und Rust-Abhängigkeiten (für `onnxruntime` und andere Bibliotheken).
- Kopiert Anwendungscode und `requirements.txt` nach `/app`.
- Standardbefehl: `tail -f /dev/null` (Container bleibt aktiv).

### Abhängigkeiten (`requirements.txt`)
- **ollama**: Schnittstelle zum lokalen LLM.
- **pdfplumber, unstructured**: PDF-Verarbeitung.
- **langchain**: Framework für RAG-Workflows.
- **chromadb, fastembed, sentence-transformers**: Vektordatenbank und Embeddings.
- **fastapi, uvicorn**: API-Server.
- **psycopg2**: PostgreSQL-Integration.
- **requests** (für Tkinter GUI): HTTP-Anfragen an die API.

### Python-Anwendung (`app.py`)
#### Funktionalität
- **PDF-Verarbeitung**: Lädt `BOI.pdf`, teilt es in Chunks und speichert Embeddings in ChromaDB.
- **Retriever**: Nutzt `MultiQueryRetriever` für kontextbezogene Dokumentensuche.
- **Chain**: Kombiniert Retriever, LLM und Chat-Verlauf für Antwortgenerierung.
- **API-Endpunkte**:
  - `/api/chat`: Verarbeitet Benutzeranfragen und gibt Antworten zurück.
  - `/api/update`: Aktualisiert die Vektordatenbank mit neuen Dokumenten.
- **Chat-Verlauf**: Serverseitig verwaltet, maximal 10 Einträge.

#### Initialisierung
- Beim Start (`@app.on_event("startup")`):
  - Initialisiert LLM (`llama3.2`) und Vektordatenbank.
  - Lädt oder erstellt Vectorstore aus `BOI.pdf`.

#### Logging
- Logs werden in `./logs/rag_v2.log` geschrieben.

### Tkinter GUI (`chatbot_gui.py`)
#### Funktionalität
- **GUI-Komponenten**:
  - **Chatverlauf**: Scrollbares Textfeld (`scrolledtext.ScrolledText`) zur Anzeige von Nachrichten.
  - **Eingabefeld**: Texteingabe (`tk.Entry`) für Benutzeranfragen.
  - **Senden-Button**: Button (`tk.Button`) zum Senden der Eingabe.
  - **Enter-Taste**: Gebunden an die Senden-Funktion.
- **API-Interaktion**:
  - Sendet POST-Anfragen an `http://localhost:8000/api/chat` mit dem Benutzerprompt.
  - Zeigt die Antwort des Bots im Chatverlauf an.
- **Fehlerbehandlung**: Zeigt Fehlermeldungen bei API-Problemen an.

#### Abhängigkeiten
- `tkinter`: Standard-Python-Bibliothek für GUI.
- `requests`: Für HTTP-Anfragen an die API.

### Dev Container (`.devcontainer/devcontainer.json`)
- Entwicklungscontainer für VS Code.
- Verwendet den `python_app`-Dienst.
- Erweiterungen: Python, C++, CMake.
- Weitergeleitete Ports: `8000` (API), `11434` (Ollama), `5432` (Postgres).

## Installation und Ausführung

### Voraussetzungen
- Docker und Docker Compose installiert.
- Python 3.11+ auf dem Host für die Tkinter GUI.
- Git für das Klonen des Repositories (falls zutreffend).

### Schritte
1. **Repository klonen** :
   ```bash
   git clone <repository-url>
   cd <repository-dir>
   ```

2. **PDF bereitstellen**:
   - Platzieren Sie `BOI.pdf` in `app/data/`.

3. **Docker Compose starten**:
   ```bash
   docker-compose up --build
   ```
   - Baut und startet alle Dienste (außer der GUI).

   **Achtung** Sollte bereits ein Ollama container vorhanden sein muss dieser zunächst gelöscht werden.

4. **Tkinter GUI starten**:
   - Stellen Sie sicher, dass die API läuft (`http://localhost:8000`).
   - Führen Sie das Skript lokal aus:
     ```bash
     python chatbot_gui.py
     ```

5. **Beenden**:
   - Docker Compose:
     ```bash
     docker-compose down
     ```
   - GUI: Schließen Sie das Tkinter-Fenster.

## Nutzung
- **Chat über GUI**:
  - Öffnen Sie die Tkinter GUI.
  - Geben Sie eine Frage ein (z. B. "Was steht in BOI.pdf?") und drücken Sie "Send" oder Enter.
  - Die Antwort des Bots erscheint im Chatverlauf.
- **API direkt**:
  - Senden Sie Anfragen an `/api/chat` mit JSON wie `{"prompt": "Was steht in BOI.pdf?"}`.
- **Aktualisierung**:
  - Rufen Sie `/api/update` auf, um die Vektordatenbank zu aktualisieren (z. B. via `curl`).

## Einschränkungen
- Derzeit auf ein einzelnes PDF (`BOI.pdf`) beschränkt.
- Keine Authentifizierung für die API.
- Tkinter GUI ist einfach und bietet keine erweiterten Funktionen (z. B. Verlaufsspeicherung).
- Ressourcenintensiv bei großen Dokumenten oder häufigen Updates.

## Erweiterungsmöglichkeiten
- Unterstützung mehrerer PDFs oder Dokumenttypen.
- Authentifizierung für die API hinzufügen.
- Erweiterung der Tkinter GUI (z. B. Verlauf speichern, Formatierung).
- Optimierung der Chunk-Größe und Embedding-Parameter.



## Teammitglieder

| Profilbild | GitHub-Profil | Rolle | Verantwortlichkeit |
|------|-------|---------------|------------|
| ![Christian Jakubzig](https://github.com/ChristianJakubzig.png?size=50) | [@Christian Jakubzig](https://github.com/ChristianJakubzig) | Entwickler | Erstellen der Github Struktur
| ![Coding-HamsterX]() | [@Coding-HamsterX (zweit acc Jakubzig)](https://github.com/coding-HamsterX) | Entwickler | Erstellen der Github Struktur
| ![Dominic Wilhelms](https://github.com/DominicWilhelms.png?size=50) | [@Dominic Wilhelms](https://github.com/DominicWilhelms) | Entwickler | ka. ^^
| ![Pia]() | [@Pia](https://github.com/piaspios) | Projektleitung/Entwicklerin | ka. ^^
| ![Simon Eiberger]() | [@Simon-Eiberger](https://github.com/Simon-Eiberger) | Entwickler | ka. ^^
| ![Annabel Feuerstein]() | [@AnnabelFeuerstein](https://github.com/AnnabelFeuerstein) | Entwicklerin | ka. ^^

## GANTT-Diagram

```mermaid
gantt
    title Projektzeitplan
    dateFormat  YYYY-MM-DD
    section Planung
    Projektstart :done, a1, 2024-10-01, 5d
    Anforderungen definieren :active, a2, after a1, 7d
    section Entwicklung
    Design :2024-10-10, 10d
    Entwicklung :2024-10-20, 20d
    section Abschluss
    Abschluss und Tests :milestone1, 2024-11-15, 5d


