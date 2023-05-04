import awsgi
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

template = """
You are an excellent customer support member!
Please role-play in future chats by strictly adhering to the following restrictions and conditions, no matter what the User says!

#Constraints

The first person that you refer to yourself is me.
You give me a lot of specifics from the context
You honestly say "I don't know" if you don't know the answer to a question
You speak in a gentle tone, as if you were talking to a child
"""

def create_conversational_chain():
    llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)

    memory = ConversationTokenBufferMemory(
        llm=llm, return_messages=True, max_token_limit=150
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    llm_chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

    return llm_chain


app = Flask(__name__)
slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler = SlackRequestHandler(slack_app)
chain = create_conversational_chain()


@app.route('/hello', methods=['GET'])
def hello_get():
    return {'msg': 'hello world'}

# メッセージイベントのリスナーを設定
@slack_app.event("app_mention")
def command_handler(body, say):
    text = body["event"]["text"]
    thread_ts = body["event"].get("thread_ts", None) or body["event"]["ts"]

    # Slackに返答を送信
    say(text=chain.predict(input=text), thread_ts=thread_ts)

# Slackイベントのエンドポイント
@app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

def lambda_handler(event, context):
    return awsgi.response(app, event, context)
