from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


pregunta = "como fijo una variable de entorno en windows sin necesidad de reiniciar y como puedo ver si esta activa?"

print("Pregunta: ", pregunta)

respuesta = llm.invoke(pregunta)

print("Respuesta: ", respuesta.content)
