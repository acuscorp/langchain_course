from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

template = "Ers un experto en marketing. Sugiere un eslogan creativo para un producto {producto}"

prompt_template = PromptTemplate(
    template=template,
    input_variables=["producto"]
)


full_prompt = prompt_template.format(producto="zapatos deportivos")

print(full_prompt)