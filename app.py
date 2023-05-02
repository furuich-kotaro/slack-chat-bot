import os

from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

from langchain.chat_models import ChatOpenAI
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
from llama_index import GPTSimpleVectorIndex, ServiceContext, NotionPageReader, GPTTreeIndex, GPTVectorStoreIndex
from llama_index.indices.composability import ComposableGraph

index_file_1 = 'notion_index.json'
index_file_2 = 'intercom_index.json'

# notion_token = os.environ["NOTION_TOKEN"]
# database_id = os.environ["NOTION_QA_DB"]
# documents = NotionPageReader(integration_token=notion_token).load_data(database_id=database_id)
# index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
# index.save_to_disk(save_path=index_file_2)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
service_context = ServiceContext.from_defaults(
    chunk_size_limit=512,
    llm_predictor=ChatGPTLLMPredictor(llm=llm)
)
notion_index = GPTSimpleVectorIndex.load_from_disk(index_file_1, service_context=service_context)
intercom_index = GPTSimpleVectorIndex.load_from_disk(index_file_2, service_context=service_context)
bot_token = os.environ["SLACK_BOT_TOKEN"]
slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]

app = Flask(__name__)
slack_app = App(token=bot_token, signing_secret=slack_signing_secret)
handler = SlackRequestHandler(slack_app)

conversations = {}

def handle_conversation(thread_ts, user_message):
    # response = notion_index.query(user_message, mode="embedding")
    graph = ComposableGraph.from_indices(
        GPTTreeIndex,
        [notion_index, intercom_index],
        index_summaries=["summary1", "summary2"]
    )
    response= graph.query(user_message)
    return str(response)
    # page_id = None
    # notion_sources = ""
    # for node_with_score in response.source_nodes:
    #     extra_info = node_with_score.node.extra_info
    #     text = node_with_score.node.text
    #     source = None
    #     if extra_info is not None and 'page_id' in extra_info:
    #         source = "https://www.notion.so/" + extra_info['page_id'].replace("-", "")

    #     source_text = f"参照情報：{source}" if source else ""
    #     notion_sources = (f"{notion_sources}\n>{text}\n>>{source_text}")
    # answer = str(response)
    # return f"{answer}\n\n{notion_sources}"

# メッセージイベントのリスナーを設定
@slack_app.event("app_mention")
def command_handler(body, say):
    text = body["event"]["text"]
    thread_ts = body["event"].get("thread_ts", None) or body["event"]["ts"]

    # ChatGPTによる応答の生成
    reply = handle_conversation(thread_ts, text)

    # Slackに返答を送信
    say(text=reply, thread_ts=thread_ts)

# Slackイベントのエンドポイント
@app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
