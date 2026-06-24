# src/ingestion.py
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
doc = fitz.open("data/Final Question Bank.pdf")
text = "".join(page.get_text() for page in doc)

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([text])

# Embed + Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore/faiss_index")

print(f"✅ {len(chunks)} chunks stored in FAISS")