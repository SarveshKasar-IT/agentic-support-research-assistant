from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE = PROJECT_ROOT / "vectorstore"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

COLLECTION_NAME = "support_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Initialize ChromaDB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTORSTORE)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ---------------------------------------------------------
# Initialize embedding model
# ---------------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ---------------------------------------------------------
# Retrieve relevant documents
# ---------------------------------------------------------

def retrieve(query: str, top_k: int = 3):
    """
    Retrieve the most relevant knowledge-base
    chunks for a user query.
    """

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]

    metadatas = results.get("metadatas", [[]])[0]

    distances = results.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "distance": distance,
            }
        )

    return retrieved


# ---------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n========================================")
    print("RAG RETRIEVAL TEST")
    print("========================================")

    query = input(
        "\nEnter your question: "
    ).strip()

    results = retrieve(query)

    print("\nRetrieved results:\n")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(f"--- Result {index} ---")

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print(
            f"Text:\n{result['text']}"
        )

        print()