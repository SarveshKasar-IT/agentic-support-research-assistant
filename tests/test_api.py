import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def api_client():
    fake_graph_module = types.ModuleType("agent.graph")
    fake_graph_module.graph = MagicMock()

    with patch.dict(sys.modules, {"agent.graph": fake_graph_module}):
        sys.modules.pop("api.main", None)
        from api.main import app
        yield TestClient(app), fake_graph_module.graph

    sys.modules.pop("api.main", None)


def test_health_endpoint(api_client):
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_endpoint(api_client):
    client, mock_graph = api_client

    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="Hello from the agent")]
    }

    response = client.post(
        "/chat",
        json={"message": "Hello", "thread_id": "test-001"},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the agent"}

    args, kwargs = mock_graph.invoke.call_args
    assert isinstance(args[0]["messages"][0], HumanMessage)
    assert args[0]["messages"][0].content == "Hello"
    assert kwargs["config"]["configurable"]["thread_id"] == "test-001"


def test_default_thread_id(api_client):
    client, mock_graph = api_client

    mock_graph.invoke.return_value = {
        "messages": [AIMessage(content="OK")]
    }

    response = client.post("/chat", json={"message": "Test"})

    assert response.status_code == 200
    assert mock_graph.invoke.call_args.kwargs["config"][
        "configurable"
    ]["thread_id"] == "default-thread"


def test_chat_content_blocks(api_client):
    client, mock_graph = api_client

    mock_graph.invoke.return_value = {
        "messages": [
            AIMessage(content=[
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ])
        ]
    }

    response = client.post("/chat", json={"message": "Test"})

    assert response.status_code == 200
    assert response.json() == {"response": "First\nSecond"}


def test_chat_missing_message(api_client):
    client, _ = api_client

    response = client.post("/chat", json={"thread_id": "test"})

    assert response.status_code == 422


def test_chat_invalid_message_type(api_client):
    client, _ = api_client

    response = client.post("/chat", json={"message": ["invalid"]})

    assert response.status_code == 422
