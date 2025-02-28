# Projektplan

Dieses Projekt ist ein RAG-System (Retrieval-Augmented Generation), das Daten aus Bibliothekskatalogen nutzt, um präzise Antworten auf Anfragen zu geben.

## Voraussetzungen

- **Docker** und **Docker Compose** müssen auf deinem System installiert sein. [Hier](https://docs.docker.com/get-docker/) findest du die Installationsanleitung.
- Ein Terminal, um die Befehle auszuführen.
- Optional VS Code um direkt im **devcontainer** zu arbeiten

## Nutzung

### 1. Docker Compose starten
Das Projekt verwendet eine `docker-compose.yml`-Datei, um die benötigten Dienste zu starten. Folge diesen Schritten:

1. Stelle sicher, dass du dich im Verzeichnis des Projekts befindest:
   ```bash
   cd /pfad/zu/deinem/projekt

2. Starte die Dienste mit
   ```bash
   docker-compose up -d

**Achtung** Sollte bereits ein Ollama container vorhanden sein muss dieser zunächst gelöscht werden.

## Teammitglieder

| Profilbild | GitHub-Profil | Rolle | Verantwortlichkeit |
|------|-------|---------------|------------|
| ![Christian Jakubzig](https://github.com/ChristianJakubzig.png?size=50) | [@Christian Jakubzig](https://github.com/ChristianJakubzig) | Entwickler | Erstellen der Github Struktur
| ![Christian Jakubzig (zweit acc)](https://github.com/ChristianJakubzig.png?size=50) | [@Christian Jakubzig (zweit acc)](https://github.com/ChristianJakubzig) | Entwickler | Erstellen der Github Struktur
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


