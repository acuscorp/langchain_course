from langchain_core.runnables import RunnableLambda, RunnableParallel, chain
from langchain_ollama import ChatOllama
import json 

# Confuguración del modelo
llm = ChatOllama(model="llama3.2", temperature=0.0)


def preprocess_text(text):
    # limpia el texto eliminando espacios extras y limitando la longitud usa .strip y limite de 500
    return text.strip()[:500]

# Convertir la funcion en un RunnableLambda
preprocessor = RunnableLambda(preprocess_text)

def generate_summary(text):
    # Genera un resumen del texto usando el modelo LLM
    prompt = f"Resume en una sola oración: {text}"
    return llm.invoke(prompt).content

sumarry_branch = RunnableLambda(generate_summary)

def analyze_sentiments(text):
    # Analiza los sentimientos del texto usando el modelo LLM
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}

    Texto: {text}"""
    response = llm.invoke(prompt).content
    try:
        # Intentar parsear la respuesta como JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # Si falla, devolver un mensaje de error
        return {"error": "No se pudo parsear la respuesta como JSON", "respuesta": response}

sentiment_branch = RunnableLambda(analyze_sentiments)

def merge_rsults(data):
    # Combina los resultados del resumen y el análisis de sentimientos en un solo diccionario
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentiment_data"]["sentimiento"],
        "razon": data["sentiment_data"]["razon"]
    }

merger = RunnableLambda(merge_rsults)


parallel_analysis = RunnableParallel({
    "resumen": sumarry_branch,
    "sentiment_data": sentiment_branch
})

# cadena completa

chain = preprocessor | parallel_analysis | merger


reviews_bath = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde.",
    "Mi dia laboral a sido muy estresante, pero logré terminar todas mis tareas a tiempo."
]

resultados_batch = chain.batch(reviews_bath)

print(resultados_batch)


