# AI-chat-bot
# 🤖 PDF Question Answering Bot

Ask questions directly from a PDF using GPT-3.5 and get accurate, context-aware answers. This project combines OpenAI, LangChain, ChromaDB, and Streamlit to create a simple chatbot interface powered by your document.

---

## 🔍 What It Does

- Loads and processes a PDF file
- Splits the content into chunks
- Generates embeddings using OpenAI
- Stores embeddings in a local Chroma vector database
- Answers user questions by retrieving relevant chunks and running them through GPT-3.5
- Streamlit provides the web interface

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install dependencies

pip install -r requirements.txt

3. Add your .env file

Create a .env file in the root directory:

OPENAI_API_KEY=sk-...

> Get your API key from: https://platform.openai.com/account/api-keys



4. Add your PDF

Put your PDF file inside the data/ directory. Make sure the filename matches the one in the code (default: TTS.pdf).

5. Run the app

streamlit run main.py


---

🧠 Tech Stack

Streamlit – UI

LangChain (modular) – Chaining logic

langchain_core, langchain_community, langchain_openai


OpenAI GPT-3.5 – LLM for answering questions

ChromaDB – Local vector database

PyMuPDF – PDF parsing

tiktoken – Token counting (used by OpenAI embeddings)



---

📁 Project Structure

.
├── main.py
├── data/
│   └── TTS.pdf
├── chroma_db/           # auto-generated vector store
├── .env
├── .gitignore
├── requirements.txt
└── README.md


---

✅ Features To Add Next

Upload multiple PDFs

Show source chunks alongside answers

Model switcher (GPT-3.5 vs GPT-4)

Deploy online via Streamlit Cloud



---

⚠ Notes

Do not commit your .env file to GitHub.

Check OpenAI usage if you hit quota issues.
