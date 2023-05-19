import csv
import os
import pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

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
print(content_template)

from langchain.docstore.document import Document

docs = []
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
        docs.append(doc)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
index = pinecone.Index("slack-bot-index")
# index.delete(delete_all=True, namespace="custom-csv-loader")

db = Pinecone.from_documents(docs, embeddings, index_name="slack-bot-index", namespace="custom-csv-loader")

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
llm = ChatOpenAI(model_name="gpt-4")

# https://github.com/hwchase17/langchain/blob/04b74d0446bdb8fc1f9e544d2f164a59bbd0df0c/docs/modules/chains/index_examples/chat_vector_db.ipynb
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)

response = qa({"question": "株式会社3rdcompassのメールアドレスは?", "chat_history": []})
print(response['answer'])
print(response['source_documents'][0].page_content)

response = qa({"question": "ブルースのワークスペースIDは?", "chat_history": []})
print(response['answer'])
print(response['source_documents'][0].page_content)

response = qa({"question": "テーブル数が3より少ない会社をおしえてください", "chat_history": []})
print(response['answer'])


response = qa({"question": "チャットツールとしてchatworkを利用している企業情報リストアップして", "chat_history": []})
print(response['answer'])

