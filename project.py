# load pdf
# split into chunks
# create embeddings
# store into chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# Local LLM
model = ChatOllama(
    model="llama3.2:3b"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

data = PyPDFLoader("document_loaders/questionBank.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

print(f"Total Chunks: {len(chunks)}")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory= "Chroma-DB"
)

print("Vector DB Created Successfully!")