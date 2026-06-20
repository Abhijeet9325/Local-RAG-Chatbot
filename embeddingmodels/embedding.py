from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

texts = (
    "Hello this is abhijit wankhade"
    "Hello this is my youtube channel"
    "Hello this is my instagram id"
)

vector = embeddings.embed_documents(texts)
print(vector)