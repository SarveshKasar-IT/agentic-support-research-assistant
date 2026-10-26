from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from agent.graph import graph


app = FastAPI(
    title="Agentic Support & Research Assistant",
    description="Agentic AI assistant with LangGraph and Gemini.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default-thread"


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.message)
            ]
        },
        config=config,
    )

    content = result["messages"][-1].content

    # Gemini may return content as a string
    # or as a list of content blocks.
    if isinstance(content, str):
        response = content

    elif isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    text_parts.append(text)

            elif isinstance(item, str):
                text_parts.append(item)

        response = "\n".join(text_parts)

    else:
        response = str(content)

    return ChatResponse(
        response=response
    )