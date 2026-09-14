from langchain_ollama import ChatOllama
from openai import chat

chat = ChatOllama(model="llama3.2", temperature=0.7)

pregunta = "como fijo una variable de entorno en windows sin necesidad de reiniciar y como puedo ver si esta activa?"

print("Pregunta: ", pregunta)

respuesta = chat.invoke(pregunta)

print("Respuesta: ", respuesta.content)




from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

chat = ChatOllama(model="llama3.2", temperature=0.7)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre. \nNombre del usuario: {nombre}\nAsistente:"
)

chain = plantilla | chat

resultado = chain.invoke({"nombre": "Noe"})

print("Resultado: ", resultado.content)