import streamlit as st
import os
import time
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader

from langchain_ollama.chat_models import ChatOllama

load_dotenv()
os.environ["USER_AGENT"] = "BankBuddyBot/1.0"
pdf_files = []  # ["banking_customers.pdf"]
file_paths = ["banking_customers.csv", "bank_kb.csv"]
persist_directory = "./chroma_db"

def get_file_timestamps(file_paths):
    return {fp: os.path.getmtime(fp) for fp in file_paths if os.path.exists(fp)}

def WebsiteLoader(urls):
    loader = WebBaseLoader(urls)
    return loader.load()

def CSVFileLoader(file_paths):
    docs = []
    for file_path in file_paths:
        loader = CSVLoader(file_path=file_path)
        docs.extend(loader.load())
    return docs

def PDFLoader(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(file_path=pdf_file)
        docs.extend(loader.load())
    return docs

def build_chroma_db(pdf_files, file_paths, persist_directory):
    pdf_list = PDFLoader(pdf_files)
    csv_list = CSVFileLoader(file_paths)
    all_documents = pdf_list + csv_list  # No flattening needed!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    splited_documents = text_splitter.split_documents(all_documents)
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    vectorstore = Chroma.from_documents(
        documents=splited_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

# ---- Timestamp Checking ---- #
if "csv_timestamps" not in st.session_state:
    st.session_state.csv_timestamps = get_file_timestamps(file_paths)
if "vectorstore" not in st.session_state:
    # First run: build DB
    st.session_state.vectorstore = build_chroma_db(pdf_files, file_paths, persist_directory)
else:
    # Check for changes
    current_timestamps = get_file_timestamps(file_paths)
    if current_timestamps != st.session_state.csv_timestamps:
        st.session_state.vectorstore = build_chroma_db(pdf_files, file_paths, persist_directory)
        st.session_state.csv_timestamps = current_timestamps
    else:
        # Load persisted DB
        embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        st.session_state.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

# ---- Streamlit UI ---- #
st.set_page_config(layout="wide")
st.title("Bank Buddy Chatbot")
st.markdown("A chatbot that can answer questions about banking customers data.")

st.sidebar.header("Settings")
with st.sidebar:
    st.header("Paste your URL")
    website_url = st.text_input("Enter URL")
MODEL = "llama3"
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

retriever = st.session_state.vectorstore.as_retriever(search_type="similarity")
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