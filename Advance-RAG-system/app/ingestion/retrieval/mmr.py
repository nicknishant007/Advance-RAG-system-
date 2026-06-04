import numpy as np

from sklearn.metrics.pairwise import (
    cosine_similarity
)


class MMRRetriever:

    @staticmethod
    def rerank(
        query_embedding,
        candidate_embeddings,
        candidate_chunks,
        top_k=8,
        lambda_param=0.5
    ):

        if not candidate_chunks:
            return []

        if len(candidate_chunks) != len(
            candidate_embeddings
        ):

            raise ValueError(
                f"MMR Error: "
                f"{len(candidate_chunks)} chunks but "
                f"{len(candidate_embeddings)} embeddings"
            )

        selected_indices = []

        similarity_to_query = (
            cosine_similarity(
                [query_embedding],
                candidate_embeddings
            )[0]
        )

        first_index = np.argmax(
            similarity_to_query
        )

        selected_indices.append(
            first_index
        )

        while len(selected_indices) < min(
            top_k,
            len(candidate_chunks)
        ):

            remaining_indices = [

                idx

                for idx in range(
                    len(candidate_chunks)
                )

                if idx not in selected_indices

            ]

            mmr_scores = []

            for idx in remaining_indices:

                relevance = (
                    similarity_to_query[idx]
                )

                diversity = max(
                    cosine_similarity(
                        [candidate_embeddings[idx]],
                        [
                            candidate_embeddings[i]
                            for i in selected_indices
                        ]
                    )[0]
                )

                mmr_score = (
                    lambda_param * relevance
                    -
                    (1 - lambda_param)
                    * diversity
                )

                mmr_scores.append(
                    (idx, mmr_score)
                )

            best_index = max(
                mmr_scores,
                key=lambda x: x[1]
            )[0]

            selected_indices.append(
                best_index
            )

        return [

            candidate_chunks[idx]

            for idx in selected_indices

        ]