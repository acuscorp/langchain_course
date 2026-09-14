import truststore
truststore.inject_into_ssl()

import bs4
from langchain_community.document_loaders import WebBaseLoader


# ============================================================
# EJEMPLO 1: Cargar una página web
# ============================================================

loader = WebBaseLoader(
    "https://docs.langchain.com/docs/"
)

docs = loader.load()

print(f"Páginas cargadas: {len(docs)}")
print(f"Título: {docs[0].metadata.get('title', 'Sin título')}")
print(f"URL: {docs[0].metadata['source']}")
print(f"Contenido:\n{docs[0].page_content[:500]}...")


# ============================================================
# EJEMPLO 2: Cargar múltiples URLs
# ============================================================

urls = [
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/how_to/"
]

loader = WebBaseLoader(
    web_paths=urls
)

docs = loader.load()

print("\n" + "=" * 60)
print("MÚLTIPLES PÁGINAS")
print("=" * 60)

print(f"Páginas cargadas: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\nPágina {i + 1}")
    print(f"URL: {doc.metadata['source']}")
    print(f"Longitud: {len(doc.page_content)} caracteres")
    print(f"Contenido: {doc.page_content[:200]}...")


# ============================================================
# EJEMPLO 3: Filtrar HTML usando BeautifulSoup
# ============================================================

loader = WebBaseLoader(
    web_paths=[
        "https://docs.langchain.com/docs/"
    ],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
             "body", {"class": ["antialiased"]}
        )
    }
)

docs = loader.load()

print("\n" + "=" * 60)
print("CONTENIDO FILTRADO")
print("=" * 60)

print(f"Páginas cargadas: {len(docs)}")
print(f"Longitud: {len(docs[0].page_content)} caracteres")
print(f"Contenido:\n{docs[0].page_content[:500]}...")