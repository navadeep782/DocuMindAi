import uuid 
import chromadb
import pytest

from app.services.vector_store import VectorStore


@pytest.fixture
def vector_store():
    client = chromadb.EphemeralClient()

    collection_name = f"test_knowledgehub_{uuid.uuid4().hex}"

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    store = VectorStore(collection)

    yield store

    client.delete_collection(name=collection_name)

def sample_chunks():
    return [
        {
            "chunk_id": 0,
            "text": "TCP provides reliable communication.",
            "source": "networks.pdf",
            "page_number": 1,
            "start_index": 0,
        },
        {
            "chunk_id": 1,
            "text": "UDP is a connectionless protocol.",
            "source": "networks.pdf",
            "page_number": 2,
            "start_index": 100,
        },
    ]


def test_add_chunks(vector_store):
    chunks = sample_chunks()

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    count = vector_store.add_chunks(
        chunks,
        embeddings,
        document_id="doc_123",
    )

    assert count == 2
    assert vector_store.collection.count() == 2


def test_search_chunks(vector_store):
    vector_store.add_chunks(
        sample_chunks(),
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        document_id="doc_123",
    )

    results = vector_store.search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == (
        "TCP provides reliable communication."
    )
    assert results[0]["metadata"]["page_number"] == 1


def test_search_by_document_id(vector_store):
    vector_store.add_chunks(
        sample_chunks(),
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        document_id="doc_123",
    )

    results = vector_store.search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=5,
        document_id="another_document",
    )

    assert results == []


def test_delete_document(vector_store):
    vector_store.add_chunks(
        sample_chunks(),
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        document_id="doc_123",
    )

    vector_store.delete_document("doc_123")

    assert vector_store.collection.count() == 0


def test_reject_mismatched_chunks_and_embeddings(vector_store):
    with pytest.raises(ValueError, match="number of chunks"):
        vector_store.add_chunks(
            sample_chunks(),
            [[1.0, 0.0, 0.0]],
            document_id="doc_123",
        )



def test_search_chunks_with_multiple_document_ids(vector_store):
    chunks = [
        {
            "chunk_id": 0,
            "text": "Atomicity ensures all operations succeed or fail together.",
            "source": "DBMS.pdf",
            "page_number": 1,
            "start_index": 0,
        },
        {
            "chunk_id": 0,
            "text": "Deadlock occurs when processes wait indefinitely.",
            "source": "OS.pdf",
            "page_number": 2,
            "start_index": 0,
        },
        {
            "chunk_id": 0,
            "text": "TCP provides reliable communication.",
            "source": "CN.pdf",
            "page_number": 3,
            "start_index": 0,
        },
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
    ]

    vector_store.add_chunks(chunks[:1], embeddings[:1], "doc_dbms")
    vector_store.add_chunks(chunks[1:2], embeddings[1:2], "doc_os")
    vector_store.add_chunks(chunks[2:], embeddings[2:], "doc_cn")

    results = vector_store.search_chunks(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=5,
        document_ids=["doc_dbms", "doc_os"],
    )

    returned_ids = {
        result["metadata"]["document_id"]
        for result in results
    }

    assert returned_ids == {"doc_dbms", "doc_os"}


def test_search_chunks_rejects_empty_document_ids(vector_store):
    import pytest

    with pytest.raises(ValueError, match="At least one document ID"):
        vector_store.search_chunks(
            query_embedding=[1.0, 0.0, 0.0],
            document_ids=[],
        )

