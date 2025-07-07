import os
from langchain_ollama import ChatOllama
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from Retriever import query_vector_store, huggingface_custom_embeddings2

# URL des Ollama-Servers
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
# Name des Sprachmodells
MODEL_NAME = "llama3.2"
# Name der Vektordatenbank
STORE_NAME = "chroma_db_huggingface_custom2"

# Konversationskette erstellen
def create_conversation_chain():
    # Prompt mit Platzhaltern für Chathistorie und Kontext
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant answering questions about Romeo and Juliet based on provided context. Use the following documents to inform your response:\n{context}\nAnswer concisely and accurately."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    # LLM für Antwortgenerierung
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)
    
    # Kontext aus Dokumenten formatieren
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Konversationskette
    chain = (
        {
            "context": lambda x: format_docs(query_vector_store(STORE_NAME, x["question"], huggingface_custom_embeddings2)),
            "question": RunnablePassthrough(),
            "history": lambda x: x["history"]
        }
        | prompt
        | llm
    )
    return chain

# Hauptfunktion für den Chatbot
def run_chatbot():
    chain = create_conversation_chain()
    history = ChatMessageHistory()
    
    print("Chatbot started. Type your question (or 'quit' to exit):")
    
    while True:
        # Benutzereingabe
        user_input = input("> ")
        if user_input.lower() == "quit":
            print("Exiting chatbot.")
            break
        
        # Nachricht zur Historie hinzufügen
        history.add_user_message(user_input)
        
        # Antwort streamen
        print("\nAssistant:")
        response = ""
        for chunk in chain.stream({"question": user_input, "history": history.messages}):
            response += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        
        # Antwort zur Historie hinzufügen
        history.add_ai_message(response)
        
        # Optional: Historie anzeigen
        print("\n--- Chat History ---")
        for msg in history.messages:
            role = "User" if msg.type == "human" else "Assistant"
            print(f"{role}: {msg.content}")
        print("\n")

if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as e:
        print(f"An error occurred: {e}")