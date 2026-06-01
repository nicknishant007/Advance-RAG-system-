
from langchain.schema import Document

from app.ingestion.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.ingestion.retrieval.mmr import (
    MMRRetriever
)

from app.ingestion.retrieval.reranker import (
    Reranker
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
        ) = self.retriever.retrieve(query)

        candidates = []

        candidate_embeddings = []

        for result in dense_results:

            text = result.payload["text"]

            metadata = result.payload[
                "metadata"
            ]

            candidates.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

            candidate_embeddings.append(
                result.vector
            )

        candidates.extend(sparse_results)

        unique_candidates = []

        seen = set()

        for chunk in candidates:

            if chunk.page_content not in seen:

                seen.add(
                    chunk.page_content
                )

                unique_candidates.append(
                    chunk
                )

        mmr_chunks = MMRRetriever.rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_chunks=unique_candidates,
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