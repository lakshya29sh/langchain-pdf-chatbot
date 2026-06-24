from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load FAISS vector store
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Load LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

while True:
    query = input("\nAsk a question (or 'quit'): ")

    if query.lower() == "quit":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only using the provided context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\nRetrieved Chunks:")
    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}:")
        print(doc.page_content[:200])