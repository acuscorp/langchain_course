import logging
import time
from operator import itemgetter
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from config import *
from prompts import *
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class LimitedMultiQueryRetriever(MultiQueryRetriever):
    max_queries: int = 3

    def generate_queries(self, question, run_manager):
        start = time.perf_counter()
        queries = super().generate_queries(question, run_manager)
        selected_queries = [query.strip() for query in queries if query.strip()][:self.max_queries]
        logger.info(
            "rag.query_variants.complete generated=%s selected=%s duration_ms=%.0f",
            len(queries),
            len(selected_queries),
            (time.perf_counter() - start) * 1000,
        )
        return selected_queries

    def retrieve_documents(self, queries, run_manager):
        start = time.perf_counter()
        documents = super().retrieve_documents(queries, run_manager)
        logger.info(
            "rag.query_variants.retrieval_complete queries=%s documents=%s duration_ms=%.0f",
            len(queries),
            len(documents),
            (time.perf_counter() - start) * 1000,
        )
        return documents


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

@st.cache_resource
def initialize_rag_system():
    start = time.perf_counter()
    logger.info(
        "rag.initialize.start chroma_path=%s query_model=%s generation_model=%s",
        CHROMA_DB_PATH,
        QUERY_LLM_MODEL,
        GENERATION_LLM_MODEL,
    )

    # Initialize the RAG system components: embeddings, LLM, vector store, and retriever
    # Vector store
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    logger.info("rag.initialize.vector_store_ready documents=%s", vector_store._collection.count())
    
    base_retriver = vector_store.as_retriever(
            search_type= RETRIEVER_SEARCH_TYPE,
            search_kwargs={
                "k": MMR_SEARCH_K,
                "lambda_mult": MMR_DIVERSITY_LAMDA,
                "fetch_k": MMR_FETCH_K,
            }
        )

    multiquery_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # multiquery retriever
    mmr_multiquery_retriever = LimitedMultiQueryRetriever.from_llm(
        retriever=base_retriver,
        prompt=multiquery_prompt,
        llm=llm_query,
        include_original=False,
    )
    mmr_multiquery_retriever.verbose = False


    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    rag_chain = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    logger.info("rag.initialize.complete duration_ms=%.0f", (time.perf_counter() - start) * 1000)
    return rag_chain, mmr_multiquery_retriever


def query_rag(question):
    trace_id = uuid4().hex[:8]
    start = time.perf_counter()
    logger.info("rag.query.start trace_id=%s question_length=%s", trace_id, len(question))

    try:
        stage_start = time.perf_counter()
        rag_chain, retriever = initialize_rag_system()
        logger.info(
            "rag.query.initialized trace_id=%s duration_ms=%.0f",
            trace_id,
            (time.perf_counter() - stage_start) * 1000,
        )

        stage_start = time.perf_counter()
        docs = retriever.invoke(question)
        logger.info(
            "rag.query.retrieval_complete trace_id=%s documents=%s duration_ms=%.0f",
            trace_id,
            len(docs),
            (time.perf_counter() - stage_start) * 1000,
        )

        stage_start = time.perf_counter()
        response = rag_chain.invoke({
            "context": format_docs(docs),
            "question": question,
        })
        logger.info(
            "rag.query.generation_complete trace_id=%s response_length=%s duration_ms=%.0f total_duration_ms=%.0f",
            trace_id,
            len(response),
            (time.perf_counter() - stage_start) * 1000,
            (time.perf_counter() - start) * 1000,
        )


        docs_info = []

        for i, doc in enumerate(docs, 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source','No source').split("/")[-1],
                "pagina": doc.metadata.get('page','No page'),
            }
            docs_info.append(doc_info)

        return response, docs_info
    except Exception as e:
        logger.exception(
            "rag.query.failed trace_id=%s duration_ms=%.0f",
            trace_id,
            (time.perf_counter() - start) * 1000,
        )
        st.error(f"Error querying RAG system ({trace_id}): {e}")
        return f"No se pudo procesar la consulta: {e}", []

def get_retriever_info():

    return {
        "tipo": f"{RETRIEVER_SEARCH_TYPE.upper()}",
        "documentos": MMR_SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMDA,
        "candidatos": MMR_FETCH_K,
        "umbral":None
    }