import os
import json
import logging
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from sentence_transformers import CrossEncoder

# Logging einrichten
logging.basicConfig(
    filename="./logs/rag_v2.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Konfiguration
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DB_PATH = "./vectorstore"
VECTOR_STORE_NAME = "simple-rag"

# FastAPI-App initialisieren
app = FastAPI(
    title="RAG-Bot API v2",
    description="Eine verbesserte API für einen Retrieval-Augmented Generation Bot mit serverseitig verwaltetem Chat-Verlauf",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic-Modell für die Eingabe
class ChatRequest(BaseModel):
    prompt: str

# Globale Variablen
llm = None
vector_db = None
retriever = None
chain = None
chat_history = []

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Schlüsselwörter für dokumentbezogene Fragen
DOC_RELATED_KEYWORDS = ["BOI", "report", "information", "procedure", "file"]

# Embedding-Funktionen
def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logger.info(f"PDF erfolgreich geladen von: {doc_path}")
        return data
    else:
        logger.error(f"PDF-Datei nicht gefunden: {doc_path}")
        return None

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Dokumente in {len(chunks)} Chunks aufgeteilt.")
    return chunks

def load_or_create_vector_db(chunks, persist_directory):
    logger.info(f"Prüfe Vectorstore an: {persist_directory}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    if os.path.exists(persist_directory):
        logger.info("Lade bestehenden Vectorstore...")
        vector_db = Chroma(
            embedding_function=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=persist_directory
        )
    else:
        if chunks is None:
            logger.error("Keine Chunks angegeben, aber Vectorstore existiert nicht!")
            raise ValueError("Keine Chunks angegeben, aber Vectorstore existiert nicht!")
        logger.info("Erstelle neuen Vectorstore...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=persist_directory
        )
    logger.info("Vectorstore erfolgreich geladen oder erstellt.")
    return vector_db

def update_vector_db(new_chunks, persist_directory):
    logger.info(f"Aktualisiere Vectorstore an: {persist_directory}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    if not os.path.exists(persist_directory):
        logger.warning("Vectorstore existiert nicht. Erstelle einen neuen.")
        vector_db = Chroma.from_documents(
            documents=new_chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=persist_directory
        )
    else:
        vector_db = Chroma(
            embedding_function=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=persist_directory
        )
        vector_db.add_documents(new_chunks)
    logger.info(f"Vectorstore mit {len(new_chunks)} neuen Chunks aktualisiert.")
    return vector_db

# Retriever und Chain erstellen
def create_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Generate five different versions of the given user question to retrieve relevant documents from a vector database. Provide these as a JSON array like ["question1", "question2", ...].
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(search_kwargs={"k": 15}),
        llm,
        prompt=QUERY_PROMPT
    )
    logger.info("Retriever created.")
    return retriever

def create_chain(retriever, llm):
    template = """You are a helpful assistant. Answer the question clearly and concisely using ONLY the following context and chat history. Use the chat history to provide context-aware answers where relevant. If the context lacks the answer, say "I don't have enough information to answer this."
    Context: {context}
    Chat History: {history}
    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "history": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("Chain created successfully.")
    return chain

# Initialisierung beim Start
@app.on_event("startup")
async def startup_event():
    global llm, vector_db, retriever, chain
    logger.info("Initialisiere LLM und Vectorstore...")
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0.1)
    if not os.path.exists(VECTOR_DB_PATH):
        data = ingest_pdf(DOC_PATH)
        if data is None:
            logger.error("Fehler beim Laden der PDF.")
            raise RuntimeError("Konnte Vectorstore nicht initialisieren.")
        chunks = split_documents(data)
        vector_db = load_or_create_vector_db(chunks, VECTOR_DB_PATH)
    else:
        vector_db = load_or_create_vector_db(chunks=None, persist_directory=VECTOR_DB_PATH)
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)
    logger.info("Initialisierung abgeschlossen.")

# API-Endpunkt für Chat mit serverseitig verwaltetem Verlauf
@app.post("/api/chat")
async def chat(request: ChatRequest):
    global chat_history
    if chain is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG-Pipeline nicht initialisiert.")

    prompt = request.prompt

    try:
        is_doc_related = any(keyword.lower() in prompt.lower() for keyword in DOC_RELATED_KEYWORDS)
        logger.info(f"Prompt: {prompt}, Is document-related: {is_doc_related}")

        if is_doc_related:
            retrieved_docs = retriever.get_relevant_documents(prompt)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            if not retrieved_docs:
                response = "I don't have enough information to answer this."
            else:
                pairs = [(prompt, doc.page_content) for doc in retrieved_docs]
                scores = cross_encoder.predict(pairs)
                ranked_docs = [retrieved_docs[i] for i in scores.argsort()[::-1][:3]]
                context = " ".join([doc.page_content for doc in ranked_docs])
                logger.info(f"Context: {context[:200]}...")
                response = chain.invoke({"question": prompt, "history": "\n".join(chat_history)})
                if prompt in response:
                    response = response.replace(prompt, "").strip()
                if "Based on the provided chat history" in response:
                    response = response.split("\n\n", 1)[-1].strip()
                if "Please note that specific requirements may vary" in response:
                    response = response.split("Please note")[0].strip()
        else:
            logger.info("Using direct LLM call for general question with history")
            history_str = "\n".join(chat_history) if chat_history else "No previous conversation."
            general_prompt = f"""You are a friendly assistant. Provide a concise answer to the question, using the chat history as context if relevant. If the history is empty or unrelated, answer the question directly.
            Chat History: {history_str}
            Question: {prompt}
            Answer:"""
            response = llm.invoke(general_prompt).content.strip()
            if not response:
                response = "I'm here and doing fine, thanks for asking!"

        text_response = response.strip()
        logger.info(f"Prompt: {prompt}, Response: {text_response}")
        chat_history.append(f"User: {prompt}")
        chat_history.append(f"Bot: {text_response}")
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return {
            "response": text_response,
            "chat_history": chat_history
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"Prompt: {prompt}, Error: {error_msg}")
        chat_history.append(f"User: {prompt}")
        chat_history.append(f"Bot: {error_msg}")
        return {
            "response": error_msg,
            "chat_history": chat_history
        }

# Endpunkt zum Aktualisieren des Vectorstores
@app.post("/api/update")
async def update_vectorstore():
    global vector_db, retriever, chain
    try:
        data = ingest_pdf(DOC_PATH)
        if data is None:
            raise HTTPException(status_code=400, detail="Fehler beim Laden der PDF.")
        chunks = split_documents(data)
        vector_db = update_vector_db(chunks, VECTOR_DB_PATH)
        retriever = create_retriever(vector_db, llm)
        chain = create_chain(retriever, llm)
        logger.info("Vectorstore erfolgreich aktualisiert.")
        return {"message": "Vectorstore updated successfully"}
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren des Vectorstores: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)