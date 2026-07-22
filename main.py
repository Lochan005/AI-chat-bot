"""
Ask Ziggy - PDF Question-Answering chatbot.

A Retrieval-Augmented Generation (RAG) app built with Streamlit and LangChain.

Two model roles:
  - Embeddings (retrieval half): runs locally with HuggingFace. No API key, no cost.
  - Generation (answer half): switchable between Groq (Llama 3) and OpenAI (GPT-4o-mini).

Run with:  streamlit run main.py
"""

import os
import hashlib

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loads GROQ_API_KEY / OPENAI_API_KEY from a local .env into the environment.
# ChatGroq and ChatOpenAI auto-read their keys from there, so we never pass
# keys around by hand.
load_dotenv()

# --- Config -----------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs on CPU
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4                          # how many chunks to retrieve per question

# The generation models the user can switch between in the sidebar.
# Each maps to the env var that must hold its API key.
PROVIDERS = {
    "Groq - Llama 3.3 70B": {
        "type": "groq",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "OpenAI - GPT-4o-mini": {
        "type": "openai",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
}

# The instruction that grounds every answer in the retrieved context and tells
# the model to refuse rather than hallucinate.
PROMPT = ChatPromptTemplate.from_template(
    """You are Ziggy, a helpful assistant that answers questions using ONLY the
context extracted from the user's PDF. If the answer is not in the context, say
you don't have enough information from the document - do not make things up.

Context:
{context}

Question: {question}

Answer:"""
)


# --- Embeddings (loaded once, reused across reruns) -------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # @st.cache_resource keeps the model weights in memory so Streamlit's
    # rerun-on-every-interaction model doesn't reload them each time.
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# --- Turn an uploaded PDF into a searchable vector store --------------------
def read_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def build_vectorstore(file_bytes: bytes) -> Chroma:
    text = read_pdf(file_bytes)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.create_documents([text])
    # In-memory Chroma (no persist_directory): the index lives only for this
    # session, so swapping PDFs can never surface stale answers from a
    # previously indexed document. This replaces the old on-disk chroma_db that
    # caused the "wrong PDF answers" class of bug.
    return Chroma.from_documents(docs, embedding=get_embeddings())


def get_vectorstore_for(file) -> Chroma:
    """Cache the built index in session state, keyed by the file's content hash,
    so re-uploading the SAME PDF reuses the index instead of re-embedding."""
    file_bytes = file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.get("pdf_hash") != file_hash:
        with st.spinner("Indexing your PDF..."):
            st.session_state.vectorstore = build_vectorstore(file_bytes)
            st.session_state.pdf_hash = file_hash
            st.session_state.messages = []   # start a fresh chat for a new doc
    return st.session_state.vectorstore


# --- Pick the generation model ----------------------------------------------
def get_llm(provider_label: str):
    cfg = PROVIDERS[provider_label]
    # Validate the key for the SELECTED provider only, with a friendly message
    # instead of a stack trace.
    if not os.getenv(cfg["key_env"]):
        st.error(
            f"Missing {cfg['key_env']}. Add it to your .env file to use "
            f"{provider_label}."
        )
        st.stop()
    if cfg["type"] == "groq":
        return ChatGroq(model=cfg["model"], temperature=0)
    return ChatOpenAI(model=cfg["model"], temperature=0)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# --- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Ask Ziggy", page_icon="dog")
st.title("Ask Ziggy about your PDF")

with st.sidebar:
    st.header("Setup")
    provider_label = st.selectbox("Generation model", list(PROVIDERS.keys()))
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if st.button("Clear chat"):
        st.session_state.messages = []
    st.caption("Embeddings run locally (HuggingFace) - no key needed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Nothing to do until a document is provided. This is why the old hardcoded
# ./data/TTS.pdf path is gone entirely - the upload replaces it.
if uploaded is None:
    st.info("Upload a PDF in the sidebar to start asking questions.")
    st.stop()

vectorstore = get_vectorstore_for(uploaded)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle a new question.
question = st.chat_input("Ask something about the document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 1) Retrieve once, then reuse the same chunks for both the answer and
        #    the Sources panel (avoids retrieving twice).
        source_docs = retriever.invoke(question)
        context = format_docs(source_docs)

        # 2) Raw LCEL chain: fill the prompt -> call the LLM -> parse to text.
        #    Built by hand instead of using create_retrieval_chain, which moved
        #    into langchain-classic in the 1.0 reorg. LCEL also streams cleanly.
        llm = get_llm(provider_label)
        chain = PROMPT | llm | StrOutputParser()

        # 3) Stream tokens into the UI as they arrive. write_stream returns the
        #    fully assembled string once streaming finishes.
        stream = chain.stream({"context": context, "question": question})
        answer = st.write_stream(stream)

        # 4) Show the retrieved chunks so every answer is auditable - a RAG
        #    best practice and a good thing to demo in an interview.
        with st.expander("Sources"):
            for i, doc in enumerate(source_docs, 1):
                st.markdown(f"**Chunk {i}**")
                st.caption(doc.page_content[:500] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
