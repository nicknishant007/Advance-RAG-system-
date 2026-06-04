import time

from difflib import (
    SequenceMatcher
)

from app.ingestion.retrieval.retrieval_pipeline import (
    RetrievalPipeline
)


class RetrievalEvaluator:

    def __init__(self):

        self.pipeline = (
            RetrievalPipeline()
        )

    def evaluate_question(
        self,
        question,
        expected_chunk,
        expected_metadata
    ):

        start_time = time.time()

        chunks = (
            self.pipeline.retrieve(
                question,
                top_k=5
            )
        )



        latency = (
            time.time()
            -
            start_time
        )

        # ----------------------------------
        # SOURCE MATCH
        # ----------------------------------

        expected_source = (
            expected_metadata[
                "source"
            ]
        )

        source_match = False

        for chunk in chunks:

            retrieved_source = (
                chunk.metadata.get(
                    "source",
                    ""
                )
            )

            if (
                retrieved_source
                ==
                expected_source
            ):

                source_match = True

                break

        # ----------------------------------
        # CHUNK SIMILARITY
        # ----------------------------------

        best_similarity = 0

        for chunk in chunks:

            similarity = (
                SequenceMatcher(
                    None,
                    expected_chunk.lower(),
                    chunk.page_content.lower()
                ).ratio()
            )

            best_similarity = max(
                best_similarity,
                similarity
            )

        # ----------------------------------
        # HIT
        # ----------------------------------

        hit = int(
            source_match
        )

        recall = hit

        # ----------------------------------
        # MRR
        # ----------------------------------

        mrr = 0

        for rank, chunk in enumerate(
            chunks,
            start=1
        ):

            if (
                chunk.metadata.get(
                    "source",
                    ""
                )
                ==
                expected_source
            ):

                mrr = (
                    1 / rank
                )

                break

        return {

            "question":
            question,

            "expected_source":
            expected_source,

            "source_match":
            source_match,

            "best_similarity":
            round(
                best_similarity,
                4
            ),

            "hit":
            hit,

            "recall":
            recall,

            "mrr":
            mrr,

            "latency":
            latency
        }