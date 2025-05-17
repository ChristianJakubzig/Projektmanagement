import os
from langchain_ollama import ChatOllama

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = "llama3.2"

model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

result = model.invoke("what is 81 divided by 9?")
print("full result:")
print(result)
print("content only:")
print(result.content)
