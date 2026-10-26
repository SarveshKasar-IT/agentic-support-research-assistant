import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from retriever import retrieve


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.8-flash"


# ---------------------------------------------------------
# Generate RAG answer
# ---------------------------------------------------------

def generate_answer(
    question: str,
    top_k: int = 3,
):

    results = retrieve(
        question,
        top_k=top_k,
    )

    context_parts = []

    for result in results:
        context_parts.append(
            result["text"]
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY
the information provided in the context below.

If the answer cannot be found in the context,
say that the knowledge base does not contain
enough information to answer the question.

Do not invent policies, procedures, prices,
dates, or other information.

Knowledge Base Context:
-----------------------
{context}
-----------------------

User Question:
{question}

Provide a clear and concise answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n========================================")
    print("RAG QUESTION ANSWERING TEST")
    print("========================================")

    question = input(
        "\nEnter your question: "
    ).strip()

    answer = generate_answer(
        question
    )

    print("\nAnswer:")
    print(answer)

    print("\n========================================")