import csv
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

# CSVファイルを開く
with open('./local/tmp/ws.csv', 'r') as f:
    reader = csv.reader(f)
    columns_list = next(reader)

columns_list = "\n".join(columns_list)

template = """
与えられたカラム名のリストから意味のある文章を作成することが目的です。
カラム名のリストは、以下のようなものです。
{columns_list}

## 条件
- 各カラム名が何を表しているのかは、カラム名から推測して考えなさい
- 文章には全てのカラム名が含まれている必要があります
- 文章は、一般的な文章として成り立っている必要があります
- 文章内のカラム名は{{}}(brace)で囲む必要があります
- カラム名をグループ化し、それぞれのグループごとに意味のある文章を作成しなさい
- write in japanaese
"""

prompt = PromptTemplate(template=template, input_variables=["columns_list"])
prompt_text = prompt.format(columns_list=columns_list)
llm = OpenAI(model_name="gpt-4")

## TODO: 質を担保する
# https://python.langchain.com/en/latest/use_cases/evaluation/data_augmented_question_answering.html
content_template=llm(prompt_text)

from langchain.docstore.document import Document
documents = []
file_path = "./local/tmp/ws.csv"
encoding = None
with open(file_path, newline="", encoding=encoding) as csvfile:
    csv_reader = csv.DictReader(csvfile)  # type: ignore
    for i, row in enumerate(csv_reader):
        content = content_template
        for k, v in row.items():
            content = content.replace(str(k), f"{k}:{v}")
        content = f"{content}\n\## row data\n{row}"
        metadata = {"source": file_path, "row_num": i }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

import uuid
from collections import Counter
from functools import partial
from transformers import BertTokenizerFast
from langchain.embeddings.openai import OpenAIEmbeddings

orig_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer = partial(
    orig_tokenizer,
    padding=True,
    truncation=True,
    max_length=512,
)

embeddings = OpenAIEmbeddings()
namespace = "hybrid-search"
batch_size = 32
texts = [ d.page_content for d in documents ]
metadatas = [ d.metadata for d in documents ]

vectors = []
ids = [str(uuid.uuid4()) for _ in texts]
for i, text in enumerate(texts):
    embedding = embeddings.embed_query(text)
    metadata = metadatas[i] if metadatas else {}
    metadata["text"] = text
    input_ids = tokenizer([text])['input_ids']
    sparse_values = {}
    for token_ids in input_ids:
        indices = []
        values = []
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        for idx in d:
            indices.append(idx)
            values.append(float(d[idx]))
        sparse_values["indices"] = indices
        sparse_values["values"] = values
    vectors.append({"id": ids[i], "values": embedding, "metadata": metadata, "sparse_values": sparse_values})

import pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("slack-bot-index")
index.upsert(vectors=vectors, namespace=namespace, batch_size=batch_size)

text = "管理者が松岡さんのワークスペースについて教えてください"
embedding = embeddings.embed_query(text)
input_ids = tokenizer([text])['input_ids']
sparse_vectors = {}
for token_ids in input_ids:
    indices = []
    values = []
    # convert the input_ids list to a dictionary of key to frequency values
    d = dict(Counter(token_ids))
    for idx in d:
        indices.append(idx)
        values.append(float(d[idx]))
    sparse_vectors["indices"] = indices
    sparse_vectors["values"] = values

query_response = index.query(
    namespace=namespace,
    top_k=3,
    include_values=True,
    include_metadata=True,
    vector=embedding,
    sparse_vectors=sparse_vectors,
)

query_response['matches'][0]['metadata']['text']
for q in query_response['matches']:
    print(q['metadata']['text'][:100])
    print("\n====================\n")

prompt_context = f"{text}\n\n## 共有情報 ##\n\n"
for q in query_response['matches']:
    prompt_context += f"{q['metadata']['text']}\n"

answer=llm(prompt_context)
print(answer)

from langchain.retrievers import PineconeHybridSearchRetriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
