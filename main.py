
import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import fitz
# Load API key from .env
# Load API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Configurations
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "gpt-3.5-turbo"

# Step 1: Load and split PDF


def load_and_split_docs():
    pdf_path = os.path.join(DATA_PATH, "TTS.pdf")
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    return docs

# Step 2: Create or load Chroma vectorstore


@st.cache_resource
def load_vectorstore():
    if os.path.exists(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    else:
        docs = load_and_split_docs()
        vectordb = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
        vectordb.persist()
        return vectordb

# Step 3: Build Retrieval QA Chain


def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name=MODEL_NAME)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 4: Streamlit UI


st.title("🐶 Ask Ziggy about your PDF")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Thinking..."):
        vectorstore = load_vectorstore()
        qa_chain = build_qa_chain(vectorstore)
        answer = qa_chain.run(user_question)
        st.success(answer)

