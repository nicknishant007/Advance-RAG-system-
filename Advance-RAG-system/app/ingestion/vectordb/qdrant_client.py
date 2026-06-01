from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance
)

from app.ingestion.vectordb.schema import (
    COLLECTION_NAME,
    VECTOR_SIZE
)


class QdrantManager:

    def __init__(self):

        self.client = QdrantClient(
            path="storage/qdrant_data"
        )

    def initialize_collection(self):

        collections = self.client.get_collections()

        existing = [
            collection.name
            for collection in collections.collections
        ]

        if COLLECTION_NAME not in existing:

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )

    def store_documents(
        self,
        ids,
        embeddings,
        chunks
    ):

        payloads = []

        for chunk in chunks:

            payloads.append({
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": idx,
                    "vector": vector.tolist(),
                    "payload": payload
                }
                for idx, vector, payload in zip(
                    ids,
                    embeddings,
                    payloads
                )
            ]

        )
    def semantic_search(
        self,
        query_vector,
        top_k=20
    ):

        return self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_vectors=True
        )