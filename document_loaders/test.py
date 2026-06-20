from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("document_loaders/questionBank.pdf")
docs = data.load()


splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 2
)

chunks = splitter.split_documents(docs)
print(len(chunks[0].page_content))
