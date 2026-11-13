
import pytest

from app.services import gemini_service


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text="ACID ensures transaction reliability."):
        self.response_text = response_text
        self.received_model = None
        self.received_prompt = None
        self.received_config = None

    def generate_content(self, model, contents, config):
        self.received_model = model
        self.received_prompt = contents
        self.received_config = config
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text="ACID ensures transaction reliability."):
        self.models = FakeModels(response_text)


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()

    monkeypatch.setattr(
        gemini_service,
        "_get_client",
        lambda: client,
    )

    return client


def test_generate_grounded_answer_success(fake_client):
    chunks = [
        {
            "text": "ACID stands for Atomicity, Consistency, Isolation and Durability.",
            "source": "DBMS_Full_Notes.pdf",
            "page_number": 12,
        }
    ]

    answer = gemini_service.generate_grounded_answer(
        question="What are ACID properties?",
        context_chunks=chunks,
    )

    assert answer == "ACID ensures transaction reliability."

    prompt = fake_client.models.received_prompt
    assert "What are ACID properties?" in prompt
    assert "DBMS_Full_Notes.pdf" in prompt
    assert "Page: 12" in prompt
    assert "Atomicity" in prompt


def test_empty_question_raises_error(fake_client):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        gemini_service.generate_grounded_answer("", [])


def test_no_context_returns_fallback_without_calling_gemini(
    monkeypatch,
):
    def unexpected_call():
        raise AssertionError("Gemini should not be called.")

    monkeypatch.setattr(
        gemini_service,
        "_get_client",
        unexpected_call,
    )

    answer = gemini_service.generate_grounded_answer(
        question="Explain an unknown topic",
        context_chunks=[],
    )

    assert "couldn't find relevant information" in answer


def test_empty_gemini_response_raises_error(monkeypatch):
    client = FakeClient(response_text="   ")

    monkeypatch.setattr(
        gemini_service,
        "_get_client",
        lambda: client,
    )

    with pytest.raises(RuntimeError, match="empty response"):
        gemini_service.generate_grounded_answer(
            question="What are ACID properties?",
            context_chunks=[
                {
                    "text": "ACID information",
                    "source": "DBMS_Full_Notes.pdf",
                    "page_number": 12,
                }
            ],
        )
