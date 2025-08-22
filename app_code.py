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
import getpass
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader


load_dotenv()


######
import getpass
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

pdf_files = ["banking_customers.pdf"]


def WebsiteLoader(urls):
    loader = WebBaseLoader(urls)
    return loader.load()

def CSVFileLoader(file_paths):
    docs = []
    for file_path in file_paths:
        loader = CSVLoader(file_path=file_path)  # Load each CSV file individually
        docs.extend(loader.load())  # Append loaded data to the list
    return docs



def PDFLoader(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(file_path=pdf_file)  # Load each PDF file individually
        docs.extend(loader.load())  # Append loaded data to the list
    return docs



pdf_list = PDFLoader(pdf_files)



llm = ChatOllama(model="llama-3.2-3b-it" )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


# Combine all loaded documents
all_documents = pdf_list

splited_documents = text_splitter.split_documents(all_documents)
#print(splited_documents)


embeddings  = OllamaEmbeddings(
  model='gte-large'
)
persist_directory = "./chroma_db"



vectorstore = Chroma.from_documents(
        documents=splited_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
print ("Using New");
vectorstore.persist()

# Step 2: Persist the database
#vectorstore.persist()  # ✅ Saves data to disk
print("✅ Data successfully stored in ChromaDB!")
###################



# ---- Streamlit UI ---- #
st.set_page_config(layout="wide")
st.title("My Local Chatbot")

st.sidebar.header("Settings")
with st.sidebar:
  st.header("Paste your URL")
  website_url = st.text_input("Enter URL")
MODEL = "llama-3.2-3b-it"
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
#embeddings = OllamaEmbeddings(model="Gte-Large")


# Initialize Chroma vector store
#vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

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
    st.session_state.chat_history.append({"role": "Banking Agent", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    #trim_memory()

    with st.chat_message("assistant"):
        response_container = st.empty()

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(prompt)
        full_response = (
            "No relevant documents found." if not retrieved_docs
            else qa({"query": prompt}).get("result", "No response generated.")
        )

        response_container.markdown(full_response)
        st.session_state.chat_history.append({"role": "banking assistant", "content": full_response})

        #trim_memory()