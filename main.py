from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# ---------------------------
# LLM
# ---------------------------
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# ---------------------------
# Embedding Model
# ---------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ---------------------------
# Load Existing ChromaDB
# ---------------------------
vector_store = Chroma(
    persist_directory="Chroma-DB",
    embedding_function=embeddings
)

# ---------------------------
# Retriever
# ---------------------------
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# ---------------------------
# Prompt Template
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Use ONLY the provided context to answer the user's question.

If the answer is not present in the context, say:

"I could not find the answer in the documents."
"""
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}
"""
    )
])

print("=" * 50)
print("RAG Chatbot Started")
print("Enter 0 to Exit")
print("=" * 50)

# ---------------------------
# Chat Loop
# ---------------------------
while True:
    query = input("\nYou: ")

    if query == "0":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    # Combine chunks into context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = model.invoke(final_prompt)

    print(f"\nAI: {response.content}")