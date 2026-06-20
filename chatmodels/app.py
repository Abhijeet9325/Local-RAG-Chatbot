from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

app = Flask(__name__)
CORS(app)

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


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")

    if not query:
        return jsonify({"error": "No message provided"}), 400

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    # Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = model.invoke(final_prompt)

    # Return sources too (for citation display)
    sources = [
        {
            "content": doc.page_content[:200],
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    return jsonify({
        "reply": response.content,
        "sources": sources,
        "source_count": len(docs)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)