# rag_chatbot/tests/test_chroma_connection.py
import pytest, requests
from rag_chatbot.config import Config

BASE = Config.CHROMA_HTTP_URL  # im Container: http://chromadb:8000

@pytest.mark.integration
def test_chroma_connection_only():
    print("BASE =", BASE)  # Debug: sollte http://chromadb:8000 sein
    hb = requests.get(f"{BASE}/api/v1/heartbeat", timeout=5)
    hb.raise_for_status()
    assert hb.status_code == 200
    print("✅ Verbindung zu Chroma steht")

@pytest.mark.integration
def test_collections():
    url = f"{BASE}/api/v1/collections"
    r = requests.get(url, params={"offset": 0, "limit": 100}, timeout=5)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # Hilfreich fürs Debugging:
        print("Status:", r.status_code)
        print("Body:", r.text)
        raise

    data = r.json()
    collections = data.get("collections", data) if isinstance(data, dict) else data
    names = [c.get("name") for c in collections if isinstance(c, dict)]

    if names:
        print("📚 Gefundene Collections:", names)
    else:
        print("ℹ️ Keine Collections vorhanden")

    assert isinstance(names, list)