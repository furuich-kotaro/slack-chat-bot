import os
import pinecone
from langchain.document_loaders import NotionDBLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_QA_DB"]
loader = NotionDBLoader(integration_token=NOTION_TOKEN, database_id=DATABASE_ID)

# 必要であれば自前で用意する
# https://github.com/hwchase17/langchain/blob/master/langchain/document_loaders/notiondb.py#L1
docs = loader.load()
for doc in docs:
    metadata = doc.metadata
    metadata = {key: str() if value is None else value for key, value in metadata.items()}
    doc.metadata = metadata

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

embeddings = OpenAIEmbeddings()
db = Pinecone.from_documents(docs, embeddings, index_name="slack-bot-index")
