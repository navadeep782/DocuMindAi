
from pathlib import Path
from functools import lru_cache

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "knowledgehub_documents"


class VectorStore:

    def __init__(self, collection):
        self.collection = collection

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str,
        file_hash: str | None = None,
    ) -> int:
        """Store document chunks, embeddings, and metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "number of chunks must match number of embeddings."
            )

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):

            if not chunk["text"].strip():
                raise ValueError("Chunk text cannot be empty.")

            if not embedding:
                raise ValueError("Embedding cannot be empty.")

            chunk_id = f"{document_id}_chunk_{chunk['chunk_id']}"

            ids.append(chunk_id)
            documents.append(chunk["text"])

            metadata = {
                "document_id": document_id,
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "start_index": chunk["start_index"],
                "chunk_id": chunk["chunk_id"],
            }

            if file_hash is not None:
                metadata["file_hash"] = file_hash

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids)

    def find_document_by_hash(self, file_hash: str) -> dict | None:
        """Find an existing document using its SHA-256 hash."""

        if not file_hash.strip():
            raise ValueError("File hash cannot be empty.")

        result = self.collection.get(
            where={"file_hash": file_hash},
            include=["metadatas"],
        )

        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]

        document_id = metadata["document_id"]

        # Ensure all returned chunks belong to the same document.
        document_records = [
            item
            for item in result["metadatas"]
            if item["document_id"] == document_id
        ]

        page_numbers = {
            item["page_number"]
            for item in document_records
        }

        return {
            "document_id": document_id,
            "filename": metadata["source"],
            "pages_with_text": len(page_numbers),
            "chunk_count": len(document_records),
            "stored_chunks": len(document_records),
            "status": "duplicate",
        }

    def update_document_file_hash(
        self,
        document_id: str,
        file_hash: str,
    ) -> int:
        """Add a SHA-256 hash to an existing document's metadata."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not file_hash.strip():
            raise ValueError("File hash cannot be empty.")

        result = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        if not result["ids"]:
            raise ValueError("Document not found.")

        updated_metadata = []

        for metadata in result["metadatas"]:
            updated_metadata.append({
                **metadata,
                "file_hash": file_hash,
            })

        self.collection.update(
            ids=result["ids"],
            metadatas=updated_metadata,
        )

        return len(result["ids"])

    def search_chunks(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        Search relevant chunks.

        Default: Search across ALL documents.
        document_id: Search one document.
        document_ids: Search selected documents.
        """

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be positive.")

        if document_id is not None and document_ids is not None:
            raise ValueError(
                "Provide document_id or document_ids, not both."
            )

        if document_id is not None and not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if document_ids is not None:

            if not document_ids:
                raise ValueError("At least one document ID is required.")

            if any(not doc_id.strip() for doc_id in document_ids):
                raise ValueError("Document IDs cannot be empty.")

            document_ids = list(dict.fromkeys(document_ids))

        total_records = self.collection.count()

        if total_records == 0:
            return []

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, total_records),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if document_id is not None:

            query_args["where"] = {
                "document_id": document_id
            }

        elif document_ids is not None:

            query_args["where"] = {
                "document_id": {
                    "$in": document_ids
                }
            }

        results = self.collection.query(**query_args)

        matches = []

        for i in range(len(results["ids"][0])):

            matches.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return matches

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        self.collection.delete(
            where={"document_id": document_id}
        )


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Create and reuse persistent ChromaDB collection."""

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return VectorStore(collection)
