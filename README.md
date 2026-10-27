
# Agentic Support & Research Assistant

An AI-powered support and research assistant built using **LangGraph, LangChain, Google Gemini, FastAPI, ChromaDB, and Docker**.

The application combines Retrieval-Augmented Generation (RAG), tool calling, web search, and conversation memory to answer questions, retrieve knowledge, perform calculations, and research external information.

## Features

- **Agentic AI:** LangGraph-based agent workflow with conditional tool execution.
- **Retrieval-Augmented Generation:** Retrieves relevant context from a ChromaDB vector database.
- **Semantic Search:** Uses Sentence Transformers (`all-MiniLM-L6-v2`) for document embeddings.
- **Web Search:** Integrates Tavily to retrieve information from the web.
- **Calculator Tool:** Safely evaluates supported arithmetic expressions.
- **Conversation Memory:** Maintains conversation context using LangGraph's `MemorySaver` and thread IDs.
- **REST API:** FastAPI endpoints for health checks and conversations.
- **Docker:** Containerized application for reproducible local execution.
- **Automated Testing:** 20 calculator and API tests integrated with GitHub Actions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| LLM | Google Gemini |
| Agent Framework | LangGraph, LangChain |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Web Search | Tavily |
| Backend API | FastAPI, Uvicorn |
| Containerization | Docker |
| Testing | Pytest, FastAPI TestClient |
| CI/CD | GitHub Actions |

## System Architecture

```text
User
  |
  v
FastAPI (/chat)
  |
  v
LangGraph Agent
  |
  +----> Google Gemini LLM
  |
  +----> RAG Retrieval
  |        |
  |        +----> Sentence Transformers
  |        |
  |        +----> ChromaDB Vector Store
  |
  +----> Tool Execution
  |        |
  |        +----> Calculator
  |        |
  |        +----> Tavily Web Search
  |
  +----> Conversation Memory (thread_id)
  |
  v
API Response
```

The application retrieves relevant knowledge-base context and supplies it to the agent. LangGraph manages model interactions and tool execution, while conversation memory preserves context for each thread.

## Project Structure

```text
agentic-support-research-assistant/
|
|-- agent/
|   |-- graph.py
|   |-- state.py
|
|-- api/
|   |-- main.py
|
|-- rag/
|   |-- ingest.py
|   |-- retriever.py
|
|-- tools/
|   |-- calculator.py
|   |-- web_search.py
|
|-- tests/
|   |-- test_calculator.py
|   |-- test_api.py
|
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|
|-- Dockerfile
|-- requirements.txt
|-- .env.example
|-- README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SarveshKasar-IT/agentic-support-research-assistant.git
cd agentic-support-research-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using `.env.example` as a reference.

Configure the Google Gemini and Tavily credentials using the environment variable names specified in `.env.example`.

Never commit actual API keys to GitHub.


## 5. Prepare the Knowledge Base

The RAG pipeline uses text documents stored in the `data/knowledge_base/` directory.

Place your `.txt` knowledge documents in this folder. For example:

```text
data/
└── knowledge_base/
    └── support_guide.txt
```

The ingestion pipeline performs the following operations:

1. Loads all `.txt` files from `data/knowledge_base/`.
2. Splits documents into chunks of 500 characters with an overlap of 100 characters.
3. Generates vector embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.
4. Stores the embeddings and source metadata in a persistent ChromaDB vector database.
5. Creates or updates the `support_knowledge` collection.

Run the ingestion pipeline from the project root:

```powershell
python -m rag.ingest
```

The generated vector database is stored in:

```text
vectorstore/
```

The application uses this knowledge base for retrieval-augmented generation (RAG), allowing the assistant to retrieve relevant document context when answering user queries.

**Note:** Run the ingestion pipeline again whenever knowledge-base documents are added or modified. Only `.txt` files in the knowledge-base directory are ingested by the current implementation.
### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

### GET /health

Checks whether the application is responding.

Example response:

```json
{
  "status": "healthy"
}
```

### POST /chat

Sends a message to the agent.

Example request:

```json
{
  "message": "Calculate 25 multiplied by 40",
  "thread_id": "demo-001"
}
```

Example response:

```json
{
  "response": "25 multiplied by 40 equals 1000."
}
```

Use the same `thread_id` for subsequent messages to maintain conversation context.

## Docker Deployment

The application can run locally inside a Docker container.

Build an image:

```bash
docker build -t agentic-support-research-assistant .
```

Run the container:

```bash
docker run -d \
  --name agentic-support-api \
  -p 8000:8000 \
  --env-file .env \
  agentic-support-research-assistant
```

The container requires valid environment variables and access to the knowledge-base/vector-store data.

**Note:** The project's locally verified image is `agentic-support-research-assistant:fixed-v3`. The commands above describe a fresh build and are not a claim that a new build has been tested.

## Automated Testing

Run the calculator and FastAPI unit tests:

```bash
python -m pytest tests/test_calculator.py tests/test_api.py -v
```

The project includes:

- 14 calculator unit tests.
- 6 FastAPI endpoint tests.
- 20 automated tests in total.

The API tests use mocking to validate endpoint behavior without requiring live LLM requests.

## Continuous Integration

GitHub Actions automatically runs the test suites on pushes and pull requests targeting the `main` branch.

Workflow file:

`.github/workflows/ci.yml`

The CI workflow uses Python 3.12 and installs the dependencies required for unit testing.

## Verified Functionality

The project has been tested locally for:

- API health checks.
- Gemini-powered conversational responses.
- Calculator tool execution.
- Knowledge-base question answering.
- Conversation memory across messages sharing a thread ID.
- Tavily web search integration.
- Docker-based API execution.
- Automated tests through GitHub Actions.

## Current Scope

The application is a working, locally containerized AI assistant with automated testing.

Public cloud hosting, authentication, persistent memory across container restarts, and production observability are outside the current verified scope.

## Repository

[GitHub — Agentic Support & Research Assistant](https://github.com/SarveshKasar-IT/agentic-support-research-assistant)
