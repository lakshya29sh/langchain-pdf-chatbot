# src/retrieval.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

def retrieve(query: str, k: int = 4):
    docs = load_vectorstore().similarity_search(query, k=k)
    return docs