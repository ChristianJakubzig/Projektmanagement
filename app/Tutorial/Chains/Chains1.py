import os
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = "llama3.2"

# Create the model with streaming enabled
model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

chain = prompt_template | model | StrOutputParser() #macht quasi das .content so das wir es nicht mehr brauchen 

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)