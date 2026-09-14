from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, neutro o negativo).")

llm = ChatOllama(model="llama3.2",temperature=0.6)

llm_estructurado = llm.with_structured_output(AnalisisTexto)

texto_prueba = "Me encanto la nueva pelicula de acción, tiene muchos efectos especiales y emocion."

resultado = llm_estructurado.invoke(f"Analiza el siguiente texto: {texto_prueba}")

# print(resultado)

# print(type(resultado))
# print(dir(resultado))
print(resultado.model_dump_json())
