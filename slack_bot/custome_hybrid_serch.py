"""Taken from: https://docs.pinecone.io/docs/hybrid-search"""
import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.retrievers import PineconeHybridSearchRetriever

def hash_text(text: str) -> str:
    return str(hashlib.sha256(text.encode("utf-8")).hexdigest())


def create_index(
    contexts: List[str],
    index: Any,
    embeddings: Embeddings,
    sparse_encoder: Any,
    ids: Optional[List[str]] = None,
) -> None:
    batch_size = 32
    _iterator = range(0, len(contexts), batch_size)
    try:
        from tqdm.auto import tqdm

        _iterator = tqdm(_iterator)
    except ImportError:
        pass

    if ids is None:
        # create unique ids using hash of the text
        ids = [hash_text(context) for context in contexts]

    for i in _iterator:
        # find end of batch
        i_end = min(i + batch_size, len(contexts))
        # extract batch
        context_batch = contexts[i:i_end]
        batch_ids = ids[i:i_end]
        # add context passages as metadata
        meta = [{"context": context} for context in context_batch]
        # create dense vectors
        dense_embeds = embeddings.embed_documents(context_batch)
        # create sparse vectors
        sparse_embeds = sparse_encoder.encode_documents(context_batch)
        for s in sparse_embeds:
            s["values"] = [float(s1) for s1 in s["values"]]

        vectors = []
        # loop through the data and create dictionaries for upserts
        for doc_id, sparse, dense, metadata in zip(
            batch_ids, sparse_embeds, dense_embeds, meta
        ):
            vectors.append(
                {
                    "id": doc_id,
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata,
                }
            )

        # upload the documents to the new hybrid index
        index.upsert(vectors)


class CustomPineconeHybridSearchRetriever(PineconeHybridSearchRetriever):
    embeddings: Embeddings
    sparse_encoder: Any
    index: Any
    top_k: int = 4
    alpha: float = 0.5
    namespace: str = ""
    query_text: str = "text"

    def get_relevant_documents(self, query: str) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale

        sparse_vec = self.sparse_encoder.encode_queries(query)
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        final_result = []
        for res in result["matches"]:
            final_result.append(Document(page_content=res["metadata"][self.query_text]))
        # return search results as json
        return final_result
