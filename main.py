import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import shutil


# --- Load API key ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Config ---
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "llama3-70b-8192"  # Groq model

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
# --- Load and split PDF ---


def load_and_split_docs():
    pdf_path = os.path.join(DATA_PATH, "Cover_Letter_GS.pdf")
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([full_text])

# --- Create/load vector DB ---


def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_PATH)
        db.persist()
        return db

# --- Build retrieval chain ---


def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(groq_api_key=groq_api_key, model=MODEL_NAME)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- UI ---


st.title("📄 Ask Your PDF (Groq-Powered)")
user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Thinking..."):
        docs = load_and_split_docs()
        vectorstore = get_vectorstore(docs)
        qa_chain = build_qa_chain(vectorstore)
        answer = qa_chain.run(user_question)
        st.success(answer)
