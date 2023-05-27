import os
import time
import logging
from datetime import datetime
from typing import List
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)
slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

def load_top_level_chat_timestamps(channel_id: str, latest_date: datetime, earliest_date: datetime):
    result_timestamps: List[int] = []
    next_cursor = None
    while True:
        try:
            result = client.conversations_history(
                channel=channel_id,
                cursor=next_cursor,
                oldest=str(earliest_date.timestamp()),
                latest=str(latest_date.timestamp()),
            )
            # TODO 招待やスレッドのチャンネルでも投稿するというメッセージがあるので、それを除外する
            result_timestamps.extend(message["ts"] for message in result["messages"])
            if not result["has_more"]:
                break
            next_cursor = result["response_metadata"]["next_cursor"]
        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                logger.error(
                    "Rate limit error reached, sleeping for: {} seconds".format(
                        e.response.headers["retry-after"]
                    )
                )
                time.sleep(int(e.response.headers["retry-after"]))
            else:
                logger.error(
                    "Error parsing conversation replies: {}".format(e))
                break
    return result_timestamps

def load_thread_chats(channel_id: str, message_ts: str):
    result_messages: List[str] = []
    next_cursor = None
    while True:
        try:
            result = client.conversations_replies(channel=channel_id, ts=message_ts, cursor=next_cursor)
            result_messages.extend(message["text"] for message in result["messages"])
            if not result["has_more"]:
                break
            next_cursor = result["response_metadata"]["next_cursor"]
        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                logger.error(
                    "Rate limit error reached, sleeping for: {} seconds".format(
                        e.response.headers["retry-after"]
                    )
                )
                time.sleep(int(e.response.headers["retry-after"]))
            else:
                logger.error(
                    "Error parsing conversation replies: {}".format(e))
    return result_messages

def get_chat_permalink(channel_id: str, message_ts: str):
    permalink = None
    try:
        res = client.chat_getPermalink(
            channel=channel_id, message_ts=message_ts)
        permalink = res["permalink"]
    except SlackApiError as e:
        if e.response["error"] == "ratelimited":
            logger.error(
                "Rate limit error reached, sleeping for: {} seconds".format(
                    e.response.headers["retry-after"]
                )
            )
            time.sleep(int(e.response.headers["retry-after"]))
        else:
            logger.error("Error parsing conversation replies: {}".format(e))
    return permalink


channel_id = "C03TZ294DJ5"
earliest_date = datetime.strptime("2023-01-1 00:00:00", '%Y-%m-%d %H:%M:%S')
latest_date = datetime.strptime("2023-05-31 00:00:00", '%Y-%m-%d %H:%M:%S')
result_timestamps = load_top_level_chat_timestamps(channel_id, latest_date, earliest_date)

from langchain.docstore.document import Document

documents = []
for message_ts in result_timestamps:
    messages = load_thread_chats(channel_id, message_ts)
    permalink = get_chat_permalink(channel_id, message_ts)
    body = "\n\n".join(messages)
    metadata = {"channel_id": channel_id, "url": permalink}
    documents.append(Document(page_content=body, metadata=metadata))

# upsert documents to pinecone
import uuid
from tqdm.auto import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder

embeddings = OpenAIEmbeddings()
# The above code is using default tfids values. It’s highly recommended to fit the tf-idf values to your own corpus.
sparse_encoder = BM25Encoder().default()
batch_size = 32
ids = [str(uuid.uuid4()) for _ in range(len(documents))]
_iterator = range(0, len(documents), batch_size)
_iterator = tqdm(_iterator)

import pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("slack-bot-index")

for i in _iterator:
    i_end = min(i + batch_size, len(documents))
    batch_ids = ids[i:i_end]
    meta = []
    context_batch = []
    for doc in documents[i:i_end]:
        metadata = doc.metadata
        encoded = doc.page_content.encode()
        truncated_bytes = encoded[:40000]
        metadata["text"] = truncated_bytes.decode(errors='ignore')
        meta.append(metadata)
        context_batch.append(doc.page_content)
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
    index.upsert(vectors=vectors, namespace="hybrid-search")

text = "国貞商店について教えて"
dense_embed = embeddings.embed_query(text)
sparse_embed = sparse_encoder._encode_single_document(text)
sparse_embed["values"] = [float(s1) for s1 in sparse_embed["values"]]
res = index.query(
    namespace="hybrid-search",
    top_k=5,
    include_values=True,
    include_metadata=True,
    vector=dense_embed,
    sparse_vectors=sparse_embed,
)

for m in res['matches']:
    print("----")
    print(m["metadata"]["text"])


from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone_text.sparse import BM25Encoder
import pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("slack-bot-index")

embeddings = OpenAIEmbeddings()
sparse_encoder = BM25Encoder().default()

text = "国貞商店について教えて"
dense_embed = embeddings.embed_query(text)
sparse_embed = sparse_encoder._encode_single_document(text)
sparse_embed["values"] = [float(s1) for s1 in sparse_embed["values"]]
res_2 = index.query(
    namespace="hybrid-search",
    top_k=5,
    include_metadata=True,
    vector=dense_embed[:200],
    sparse_vectors=sparse_embed,
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    max_tokens_limit=8192
)
response = qa({"question": "国貞商店について教えて", "chat_history": []})
sources = []
docs = response['source_documents']
for doc in docs:
    if 'url' in doc.metadata:
        sources.append(doc.metadata['url'])
    # else:
    #     url = f"https://www.notion.so/{doc.metadata['id']}"
    #     sources.append(url)


