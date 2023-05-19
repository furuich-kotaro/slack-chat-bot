import os
import json
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

QA_INDEX_NAME = "slack-bot-index"
index = pinecone.Index(QA_INDEX_NAME)
embeddings = OpenAIEmbeddings()
db = Pinecone(index, embeddings.embed_query, "text", namespace="custom-csv-loader")
llm = ChatOpenAI(model_name="gpt-4")

# https://github.com/hwchase17/langchain/blob/04b74d0446bdb8fc1f9e544d2f164a59bbd0df0c/docs/modules/chains/index_examples/chat_vector_db.ipynb
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)
chat_history = {}

def generate_answer_reply(text : str, thread_ts : int):
    if thread_ts not in chat_history:
        chat_history[thread_ts] = [(text, '')]

    response = qa({"question": text, "chat_history": chat_history[thread_ts]})
    return response["answer"]
    answer = response["answer"]
    if thread_ts in chat_history:
        chat_history[thread_ts].append((text, answer))

    sources = []
    docs = response['source_documents']
    for doc in docs:
        if 'url' in doc.metadata:
            sources.append(doc.metadata['url'])
        else:
            url = f"https://www.notion.so/{doc.metadata['id']}"
            sources.append(url)

    sources = "\n・".join(sources)
    return f"{answer}\n\n■参照情報\n・{sources}"

from slack_bolt import App
from flask import Flask, request
from slack_bolt.adapter.flask import SlackRequestHandler

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

flask_app = Flask(__name__)
app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
    process_before_response=True
)
slack_handler = SlackRequestHandler(app)

@app.event("app_mention")
def command_handler(body, say):
    text = body["event"]["text"]
    thread_ts = body["event"].get("thread_ts", None) or body["event"]["ts"]
    answer = generate_answer_reply(text=text, thread_ts=thread_ts)
    say(text = answer, thread_ts=thread_ts)

@flask_app.route('/hello', methods=['GET'])
def hello_get():
    return { "statusCode": 200, "body": json.dumps({ 'message': "hello world" }) }

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return slack_handler.handle(request)

import awsgi

# TODO 非同期処理にして200レスポンスを先に返せるようにする
def lambda_handler(event, context):
    headers = event["headers"]
    retry_num = headers.get("X-Slack-Retry-Num", None)
    retry_reason = headers.get("X-Slack-Retry-Reason", None)
    if retry_num and retry_reason == "http_timeout":
        return {
            "statusCode": 200,
            "headers": { "Content-Type": "application/json" },
            "body": { "message ": "No need to resend" }
        }
    else:
        return awsgi.response(flask_app, event, context)
