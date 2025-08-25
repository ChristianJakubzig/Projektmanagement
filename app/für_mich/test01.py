from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="https://ollama-bim24.apps.rhos.th-wildau.de",
    model="llama3.2",
    keep_alive="5m",
    temperature=0.7,
)

for chunk in llm.stream("Erzähl mir eine geschichte über Pandas"):
    print(chunk.content, end="", flush=True)