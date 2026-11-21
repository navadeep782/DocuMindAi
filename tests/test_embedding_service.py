
import pytest

from app.services import embedding_service


class FakeEmbeddingModel:

    def encode(self, texts, **kwargs):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


def test_embed_texts_returns_vectors(monkeypatch):

    monkeypatch.setattr(
        embedding_service,
        "_get_model",
        lambda: FakeEmbeddingModel(),
    )

    vectors = embedding_service.embed_texts(
        ["TCP is reliable.", "UDP is connectionless."]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


def test_embed_texts_empty_input():

    result = embedding_service.embed_texts([])

    assert result == []


def test_embed_texts_rejects_blank_text():

    with pytest.raises(ValueError, match="empty"):

        embedding_service.embed_texts(
            ["TCP is reliable.", "   "]
        )
