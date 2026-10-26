from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"
VECTORSTORE = PROJECT_ROOT / "vectorstore"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

COLLECTION_NAME = "support_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------

def load_documents():
    documents = []

    for file_path in KNOWLEDGE_BASE.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


# ---------------------------------------------------------
# Split documents into chunks
# ---------------------------------------------------------

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = []

    for document in documents:
        split_texts = splitter.split_text(document["text"])

        for index, chunk in enumerate(split_texts):
            chunks.append(
                {
                    "id": f"{document['source']}_{index}",
                    "text": chunk,
                    "source": document["source"],
                }
            )

    return chunks


# ---------------------------------------------------------
# Create vector database
# ---------------------------------------------------------

def build_vector_database(chunks):
    VECTORSTORE.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
    ).tolist()

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk["source"]
            }
            for chunk in chunks
        ],
    )

    return collection


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print("\n========================================")
    print("RAG INGESTION PIPELINE")
    print("========================================")

    print("\n1. Loading documents...")

    documents = load_documents()

    print(f"Loaded documents: {len(documents)}")

    if not documents:
        raise RuntimeError(
            "No .txt documents found in data/knowledge_base"
        )

    print("\n2. Creating text chunks...")

    chunks = create_chunks(documents)

    print(f"Created chunks: {len(chunks)}")

    print("\n3. Creating embeddings and storing vectors...")

    collection = build_vector_database(chunks)

    print("\n4. Vector database ready!")

    print(f"Collection: {collection.name}")
    print(f"Total vectors: {collection.count()}")

    print("\n========================================")
    print("RAG INGESTION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
