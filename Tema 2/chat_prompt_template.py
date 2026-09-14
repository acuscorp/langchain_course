from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOllama(model="llama3.2", temperature=0.7)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del español al inglés muy preciso y profesional."),
    ("human", "{texto}")       
])

chain = chat_prompt | chat

print(chain.invoke({"texto": "Hola, ¿cómo estás? Quiero saber como se dice en ingles 'necesito ayuda'"}).content)    



# manda el mesaje usando el prompt al modelo LLM

