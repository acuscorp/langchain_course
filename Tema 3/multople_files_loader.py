import truststore
truststore.inject_into_ssl()

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os


# Cargar todos los archivos Markdown del proyecto
loader = DirectoryLoader(
    r"\\wsl.localhost\Ubuntu-20.04\home\noe\solera-agentic-flow",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    recursive=True,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()

print(f"Documentos cargados: {len(docs)}")


# Análisis del contenido cargado
total_chars = sum(len(doc.page_content) for doc in docs)

file_stats = {}

for doc in docs:
    filename = os.path.basename(doc.metadata["source"])

    file_stats[filename] = {
        "chars": len(doc.page_content),
        "words": len(doc.page_content.split()),
        "lines": doc.page_content.count("\n") + 1
    }


# Mostrar estadísticas
print(f"\nTotal de caracteres procesados: {total_chars:,}")

print("\nTop 5 archivos más largos:")

sorted_files = sorted(
    file_stats.items(),
    key=lambda x: x[1]["chars"],
    reverse=True
)

for filename, stats in sorted_files[:5]:
    print(
        f"  {filename}: "
        f"{stats['chars']:,} chars, "
        f"{stats['words']:,} words"
    )