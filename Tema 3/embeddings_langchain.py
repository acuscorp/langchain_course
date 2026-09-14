from langchain_ollama import OllamaEmbeddings
import numpy as np

embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)


texto1 = "Paris es un nombre comun para perros"

texto2 = "Paris es la ciudad capital de Francia"

texto3 = "La capital de Francia es Paris"


vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)
vec3 = embeddings.embed_query(texto3)


print(f"Dimension del vector 1: {len(vec1)}")
print(f"Dimension del vector 2: {len(vec2)}")
print(f"Dimension del vector 3: {len(vec3)}")

cos_sim = np.dot(vec3, vec2) / (np.linalg.norm(vec3) * np.linalg.norm(vec2))
print(f"Similitud coseno entre vector 1 y vector 2: {cos_sim:.3f}")