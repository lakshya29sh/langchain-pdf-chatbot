# src/chatbot.py
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from .env

def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()
    query = "What is the passing criteria?"
    result = chain.invoke({"query": query})
    print("\n🤖 Answer:", result["result"])