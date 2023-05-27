import os
import pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("slack-bot-index")

from collections import Counter
from functools import partial
from transformers import BertTokenizerFast
from typing import List, Dict, Any, Callable
from langchain.embeddings.openai import OpenAIEmbeddings

def build_dict(input_batch: List[List[int]]) -> List[Dict[str, Any]]:
    sparse_emb = []
    # iterate through input batch
    for token_ids in input_batch:
        indices = []
        values = []
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        for idx in d:
            indices.append(idx)
            values.append(float(d[idx]))
        sparse_emb.append({"indices": indices, "values": values})
    # return sparse_emb list
    return sparse_emb

def generate_sparse_vectors(
    context_batch: List[str], tokenizer: Callable
) -> List[Dict[str, Any]]:
    inputs = tokenizer(context_batch)["input_ids"]
    sparse_embeds = build_dict(inputs)
    return sparse_embeds

def hybrid_score_norm(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    return [v * alpha for v in dense], hs

orig_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer = partial(
    orig_tokenizer,
    padding=True,
    truncation=True,
    max_length=512,
)
embeddings = OpenAIEmbeddings()

text = "国貞商店の企業情報を教えて"
dense = embeddings.embed_query(text)
sparse = generate_sparse_vectors([text], tokenizer)

hdense, hsparse = hybrid_score_norm(dense, sparse[0], alpha=0.99)
res_1 = index.query(
    namespace="hybrid-search",
    top_k=2,
    include_metadata=True,
    vector=hdense,
    sparse_vectors=hsparse,
)


res_2 = index.query(
    namespace="hybrid-search",
    top_k=2,
    include_metadata=True,
    vector=dense,
)


from collections import Counter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
