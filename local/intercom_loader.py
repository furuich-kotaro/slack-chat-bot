import os
import pinecone
from bs4 import BeautifulSoup
from llama_index import download_loader
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone

intercom_access_token = os.environ["INTERCOM_TOKEN"]
IntercomReader = download_loader("IntercomReader")
loader = IntercomReader(intercom_access_token=intercom_access_token)
articles = loader.get_all_articles()

docs = []
for article in articles:
    body = article['body']
    soup = BeautifulSoup(body, 'html.parser')
    body = soup.get_text()
    body = f"要約\n{article['description']}\n\n■本文\n{body}\n\n■記事URL\n{article['url']}"
    metadata = {
        "id": article['id'],
        "title": article['title'],
        "url": article['url'],
        'workspace_id': article['workspace_id'],
        'parent_id': article['parent_id'],
        'parent_type': article['parent_type'],
    }
    metadata = {key: str() if value is None else value for key, value in metadata.items()}
    docs.append(Document(page_content=body, metadata=metadata))

embeddings = OpenAIEmbeddings()
db = Pinecone.from_documents(docs, embeddings, index_name="slack-bot-index")
