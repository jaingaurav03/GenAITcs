import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, Milvus, MongoDBAtlasVectorSearch, ElasticVectorSearch

import os
from dotenv import load_dotenv

load_dotenv()


# ---- Streamlit UI ---- #
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

st.sidebar.header("Settings")
MODEL = st.sidebar.selectbox("Choose a Model", ["llama3.2"], index=0)
MAX_HISTORY = st.sidebar.number_input("Max History", 1, 10, 2)
CONTEXT_SIZE = st.sidebar.number_input("Context Size", 1024, 16384, 8192, step=1024)

# ---- Session State Setup ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state or st.session_state.get("prev_context_size") != CONTEXT_SIZE:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.prev_context_size = CONTEXT_SIZE

# ---- LangChain Components ---- #
llm = ChatOllama(model=MODEL, streaming=True)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# Initialize Chroma vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---- Trim Chat Memory ---- #
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)  # Remove oldest messages


# ---- Handle User Input ---- #
if prompt := st.chat_input("Say something"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(prompt)
        full_response = (
            "No relevant documents found." if not retrieved_docs
            else qa({"query": prompt}).get("result", "No response generated.")
        )

        response_container.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        trim_memory()