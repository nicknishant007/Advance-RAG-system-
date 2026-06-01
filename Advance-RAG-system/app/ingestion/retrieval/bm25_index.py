import os
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class BM25Indexer:

    def __init__(self):

        self.documents = []

        self.tokenized_docs = []

        self.bm25 = None

        # storage path
        self.storage_path = (
            
            BASE_DIR / "storage" / "bm25" / "bm25_store.pkl"
        )

        # create folder if not exists
        os.makedirs(
            os.path.dirname(self.storage_path),
            exist_ok=True
        )

        # auto load existing bm25
        self.load()

    def add_documents(self, chunks):

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        self.documents.extend(chunks)

        self.tokenized_docs.extend([
            text.lower().split()
            for text in texts
        ])

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

        # save after adding docs
        self.save()

    def search(
        self,
        query,
        top_k=10
    ):

        if self.bm25 is None:
            return []

        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            chunk
            for chunk, _ in ranked[:top_k]
        ]

    def save(self):

        data = {
            "documents": self.documents,
            "tokenized_docs": self.tokenized_docs
        }

        with open(
            self.storage_path,
            "wb"
        ) as f:

            pickle.dump(data, f)

    def load(self):

        if not os.path.exists(
            self.storage_path
        ):
            return

        with open(
            self.storage_path,
            "rb"
        ) as f:

            data = pickle.load(f)

        self.documents = data[
            "documents"
        ]

        self.tokenized_docs = data[
            "tokenized_docs"
        ]

        if self.tokenized_docs:

            self.bm25 = BM25Okapi(
                self.tokenized_docs
            )