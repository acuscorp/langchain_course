
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


#  instanciate the embeddings model
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)


# create database
vector_store = Chroma(embedding_function=embeddings, persist_directory="/home/noe/curso_langchain/Tema 3/chroma_db")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# requestor query example
# query = "¿Cuál es el inmueble del contrato en el que participa María Ximenes Campos?"
query = "Dónde se encuentra el inmueble en el que participa María Ximenes Campos?"
results = retriever.invoke(query)

print("="*50)
print("Query results:")
print("="*50)

for i, result in enumerate(results):
    print(f"Result {i+1}: {result.page_content}")
    print("-" * 50)
    print(f"Metadata: {result.metadata}")
    print("=" * 50)
    print()



