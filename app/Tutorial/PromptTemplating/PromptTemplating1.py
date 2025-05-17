
from langchain.prompts import ChatPromptTemplate

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from Template ----")
prompt = prompt_template.invoke({"topic" : "cats"})
print(prompt)