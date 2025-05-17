import os
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = "llama3.2"

# Create the model with streaming enabled
model = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)

#Part 1: create a ChatPromptTemplate using a template string
print("----Prompt from Template----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)

#Part 2: Prompt with multiple Placeholders
print("\n---- Prompt with Multiple Placeholders ----")
template_multiple = """you are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

result = model.invoke(prompt)
print(result.content)

print("\n---- Prompt with System and Human Messages (Tuple) ----")
messages = [
    {"role": "system", "content": "You are a comedian who tells jokes about {topic}."},
    {"role": "human", "content": "Tell me {joke_count} jokes."},
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)