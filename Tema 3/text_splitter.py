import pymupdf

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. CARGAR PDF
# ============================================================

pdf_path = "/home/noe/curso_langchain/Tema 3/quijote.pdf"

docs = []

with pymupdf.open(pdf_path) as pdf:

    for page_number, page in enumerate(pdf):

        text = page.get_text().strip()

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_number + 1
                    }
                )
            )


print(f"Documentos cargados: {len(docs)}")


# ============================================================
# 2. PREPARAR EL TEXTO
# ============================================================

document_text = "\n\n".join(
    f"Página {doc.metadata['page']}:\n{doc.page_content}"
    for doc in docs
)

print(f"Caracteres totales: {len(document_text):,}")


# ============================================================
# 3. DIVIDIR EL DOCUMENTO EN CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(f"Chunks generados: {len(chunks)}")

print("\nPrimer chunk:")
print(chunks[0].page_content[:500])

print("\nMetadata del primer chunk:")
print(chunks[0].metadata)


# ============================================================
# 4. CREAR LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)

# llm = ChatOllama(
#     model="qwen2.5:7b",
#     temperature=0.2
# )

# ============================================================
# 5. RESUMIR
# ============================================================



summaries = []
cnt=0
for doc in docs:
    if cnt > 10:
        break
    response = llm.invoke(f"Haz un resumen de los puntos más importantes del siguiente texto {doc}")
    summaries.append(response)
    cnt+=1


summary=llm.invoke(f"Combina y sintetiza estos resumentes en un resument completo: {summaries}")


# ============================================================
# 6. MOSTRAR RESULTADO
# ============================================================

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(summary.content)