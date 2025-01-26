from dotenv import load_dotenv
from flask import Flask
from chatbot_api.sessions import storage
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI()

### Construct retriever ###
data_path = "./chatbot_api/data/llm-data.txt"
loader = TextLoader(data_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

### Contextualize chat history ###
contextualize_history_system_prompt = (
    "Given a chat history and the latest user question and answer "
    "which might reference context in the chat history, "
    "keep the chat history in mind and consider it for the future response."
)

contextualize_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_history_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_history_prompt
)

### Giving response ###
system_prompt = (
    "You are a chatbot assistant named AdaptiveBot that could answer question related to Adaptive Network Laboratory resources."
    "You have to answer user's question in Bahasa Indonesia but keep computer-science terms in English such as 'server'."
    "Use the following pieces of retrieved context to answer question from users that are the research assistants of Adaptive Network Laboratory."
    "Make the conversation as concise as possible."
    "Please use simple words to ensure user understand you."
    "If the user asks something that you don't know, please refuse to answer their question by saying something like 'Sorry, aku ga tau bro!'"
    "\n\n"
    "{context}"
)

compiled_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

conversation_chain = create_stuff_documents_chain(llm, compiled_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, conversation_chain)

### Statefully manage chat history ###
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in storage:
        storage[session_id] = ChatMessageHistory()
    return storage[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

app = Flask(__name__)

# Initialize API and register all routes
from flask_restful import Api
api = Api(app)
from chatbot_api.routes import register_routes
register_routes(api=api)