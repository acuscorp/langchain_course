from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("C:\\Users\\Noe.Acuna\\curso_langchain\\Tema 3\\Noe Adrian.pdf")
pages = pdf_loader.load()
print(dir(pages[0]))
print(f"\nPDF Title --- {pages[0].page_content.title}")
print(f"\nPDF Content --- {pages[0].page_content}")
