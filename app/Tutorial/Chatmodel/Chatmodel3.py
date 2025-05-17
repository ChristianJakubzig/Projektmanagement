import os
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = "llama3.2"

# Create the model with streaming enabled
model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query = input("you: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    # Stream the response
    print("AI: ", end="", flush=True)
    response_content = ""
    for chunk in model.stream(chat_history):
        chunk_content = chunk.content
        print(chunk_content, end="", flush=True)
        response_content += chunk_content
    
    print()  # Add a newline after the response
    
    # Add the complete response to history
    chat_history.append(AIMessage(content=response_content))

print("---- Message History ----")
print(chat_history)