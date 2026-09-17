from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from config import *
from prompts import *
import streamlit as st

@st.cache_resource
def initialize_rag_system():
    # Initialize the RAG system components: embeddings, LLM, vector store, and retriever
    # Vector store
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    base_retriver = vector_store.as_retriever(
            search_type= RETRIEVER_SEARCH_TYPE,
            search_kwargs={
                "k": MMR_SEARCH_K,
                "lamda_mult": MMR_DIVERSITY_LAMDA,
                "fetch_k": MMR_FETCH_K,
            }
        )

    multiquery_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # multiquery retriever
    mmr_multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriver,
        prompt=multiquery_prompt,
        llm=llm_query
    )


    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Function to format and preprocess the input for the RAG chain 
    def format_docs(docs):
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            header = f"[Fragmetn {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("/")[-1] if "/" in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Source: ({source})"
                if 'page' in doc.metadata:
                    page = doc.metadata['page']
                    header += f" - Page: ({page})"
            
            content = doc.page_content.strip()
            formatted_docs.append(f"{header}\n{content}")
        return "\n\n".join(formatted_docs)

    rag_chain = (
        {
            "context": mmr_multiquery_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multiquery_retriever


def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        response = rag_chain.invoke(question)

        docs = retriever.get_relevant_documents(question)


        docs_info = []

        for i, doc in enumerate(docs, 1):
            doc_info = {
                "fragment": i,
                "content": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "source": doc.metadata.get('source','No source').split("/")[-1],
                "page": doc.metadata.get('page','No page'),
            }
            docs_info.append(doc_info)

        return response, docs_info
    except Exception as e:
        st.error(f"Error querying RAG system: {e}")
        print(f"Error querying RAG system: {e}")
        return None, []

def get_retriever_info():

    return {
        "tipo": f"{RETRIEVER_SEARCH_TYPE.upper()}",
        "documentos": MMR_SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMDA,
        "candidatos": MMR_FETCH_K,
        "umbral":None
    }