
from langchain_core.documents import (Document)

from app.ingestion.retrieval.hybrid_retrieval import (
    HybridRetriever
)

from app.ingestion.retrieval.mmr import (
    MMRRetriever
)

from app.ingestion.retrieval.reranker import (
    Reranker
)
from app.ingestion.embedding.embedder import (
    Embedder
)


class RetrievalPipeline:

    def __init__(self):

        self.retriever = HybridRetriever()

    def retrieve(
        self,
        query,
        top_k=5
    ):

        (
            query_embedding,
            dense_results,
            sparse_results
        ) = self.retriever.semantic_search(query)

        candidates = []

        candidate_embeddings = []

        seen = set()

        # Dense Retrieval Results
        for result in dense_results:

            text = result.payload["text"]

            metadata = result.payload[
                "metadata"
            ]

            if text not in seen:

                seen.add(text)

                candidates.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

                candidate_embeddings.append(
                    result.vector
                )

        # Sparse Retrieval Results
        for chunk in sparse_results:

            if chunk.page_content not in seen:

                seen.add(
                    chunk.page_content
                )

                candidates.append(chunk)

                sparse_embedding = Embedder.embed(
                    [chunk.page_content]
                )[0]

                candidate_embeddings.append(
                    sparse_embedding
                )

        mmr_chunks = MMRRetriever.rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_chunks=candidates,
            top_k=10
        )

        reranked_chunks = (
            Reranker.rerank(
                query=query,
                chunks=mmr_chunks,
                top_k=top_k
            )
        )

        return reranked_chunks