import requests
from typing import List, Dict, Any
from ..config import Config
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.embedding_model = Config.OLLAMA_EMBEDDING_MODEL
        self.keep_alive = Config.OLLAMA_KEEP_ALIVE

    