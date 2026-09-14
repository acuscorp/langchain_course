from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

#  instanciate the embeddings model
EMBEDDING_MODEL = "bge-m3"
BASE_URL = "http://localhost:11434"
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=BASE_URL,
)


#  instanciate the LLM model
QUERY_LLM_MODEL = "llama3.2"
GENERATION_LLM_MODEL="qwen2.5"
TEMPERATURE=0.2

llm_query = ChatOllama(
    model=QUERY_LLM_MODEL,
    temperature=TEMPERATURE
)

llm_generation = ChatOllama(
    model=GENERATION_LLM_MODEL,
    temperature=TEMPERATURE
)

CHROMA_DB_PATH="/home/noe/curso_langchain/Tema 3/chroma_db"

# Retriever configuration
RETRIEVER_SEARCH_TYPE = "mmr"

# mmr (Maximal Margin Relevance) configuration
MMR_DIVERSITY_LAMDA = 0.7
MMR_FETCH_K = 20
MMR_SEARCH_K = 2


