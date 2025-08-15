# rag_chatbot/tests/test_chroma_connection.py
import pytest
import requests
from rag_chatbot.config import Config

BASE = Config.CHROMA_HTTP_URL  # z.B. http://chromadb:8000

@pytest.mark.integration
def test_chroma_connection():
    """Prüft, ob ChromaDB v2 erreichbar ist."""
    url = f"{BASE}/api/v2/version"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        version = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        print(f"✅ ChromaDB erreichbar – Version: {version}")
        assert r.status_code == 200
    except requests.RequestException as e:
        pytest.fail(f"❌ Verbindung zu ChromaDB fehlgeschlagen: {e}")


@pytest.mark.integration
def test_collections():
    """Prüft, ob Collections vorhanden sind und listet sie."""
    tenant = "default_tenant"
    database = "default_database"

    # Tenant & Database ermitteln
    try:
        identity_url = f"{BASE}/api/v2/auth/identity"
        identity_response = requests.get(identity_url, timeout=5)
        if identity_response.status_code == 200:
            identity_data = identity_response.json()
            tenant = identity_data.get("tenant", tenant)
            databases = identity_data.get("databases", [database])
            if databases:
                database = databases[0]
    except requests.RequestException:
        print("⚠️ Identity-Endpoint nicht verfügbar, verwende Default-Werte")

    collections_url = f"{BASE}/api/v2/tenants/{tenant}/databases/{database}/collections"

    try:
        r = requests.get(collections_url, timeout=5)
        r.raise_for_status()
        data = r.json()
        collections = data if isinstance(data, list) else []
        names = [c.get("name", c.get("id")) for c in collections if isinstance(c, dict)]

        if names:
            print("📚 Gefundene Collections:", names)
        else:
            print("ℹ️ Keine Collections vorhanden")

        assert isinstance(collections, list)
    except requests.RequestException as e:
        pytest.fail(f"❌ Fehler beim Abrufen der Collections: {e}")
