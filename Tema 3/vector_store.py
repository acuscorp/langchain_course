
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# instanciate the pdf dir loader

pdf_loader = PyPDFDirectoryLoader("/home/noe/curso_langchain/Tema 3/contratos")
documents = pdf_loader.load()

# instanciate the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents)

# instantiate the embeddings model

print(f"Documents have been split into chunks. Total chunks: {len(split_documents)}")

#  instanciate the embeddings model
embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)


# create database
vector_store = Chroma.from_documents(split_documents, embeddings,persist_directory="/home/noe/curso_langchain/Tema 3/chroma_db")

print("Vector store has been created successfully.")

# requestor query example
# query = "¿Cuál es el inmueble del contrato en el que participa María Ximenes Campos?"
query = "Dónde se encuentra el inmueble en el que participa María Ximenes Campos?"
results = vector_store.similarity_search(query,k=2)

print("="*50)
print("Query results:")
print("="*50)

for i, result in enumerate(results):
    print(f"Result {i+1}: {result.page_content}")
    print("-" * 50)
    print(f"Metadata: {result.metadata}")
    print("=" * 50)
    print()



