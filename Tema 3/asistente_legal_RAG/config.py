from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DB_PATH = BASE_DIR.parent  / "chroma_db"

print("Chroma path:", CHROMA_DB_PATH)


#  instanciate the embeddings model
EMBEDDING_MODEL = "bge-m3"
BASE_URL = "http://localhost:11434"
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=BASE_URL,
)


#  instanciate the LLM model
QUERY_LLM_MODEL = "llama3.2"
GENERATION_LLM_MODEL="qwen2.5:7b"
TEMPERATURE=0.2

llm_query = ChatOllama(
    model=QUERY_LLM_MODEL,
    temperature=TEMPERATURE,
    base_url=BASE_URL,
)

llm_generation = ChatOllama(
    model=GENERATION_LLM_MODEL,
    temperature=TEMPERATURE,
    base_url=BASE_URL,
)


# Retriever configuration
RETRIEVER_SEARCH_TYPE = "mmr"

# mmr (Maximal Margin Relevance) configuration
MMR_DIVERSITY_LAMDA = 0.7
MMR_FETCH_K = 20
MMR_SEARCH_K = 2