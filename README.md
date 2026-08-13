# Local RAG-Based PDF Chatbot

A fully local, zero-cost Retrieval-Augmented Generation (RAG) chatbot that allows you to upload PDF documents and chat with them in real-time. 

Built using **LangChain** (orchestration framework), **FAISS** (local vector database), **Hugging Face** (provides the embedding model), **Ollama (Llama 3)** (Local LLM Engine), and a customizable **Streamlit** UI.
---
## Features

* **100% Free & Local:** Runs completely offline on your own hardware.
* **Semantic Search & Retrieval:** Uses dense vector embeddings to find the most relevant context blocks from your PDFs before generating an answer.
* **Source Citations:** Every response includes an expandable section displaying the precise source text and page numbers used to generate the answer.
* **Efficient Caching:** Embeddings and vector stores are handled in session memory for fast, seamless queries.
---
## Tech Stack

* **Orchestration:** LangChain (`langchain-classic`, `langchain-community`)
* **Embeddings:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search) running locally in-memory
* **LLM Engine:** Ollama running open-source models like `llama3`
* **Frontend UI:** Streamlit with custom CSS styling
---
## Use the Repo:

### Prerequisites

Make sure you have Python (3.9 or higher) installed on your system. You also need **Ollama** installed to run the local LLM.

1. **Download & Install Ollama:**
   Get it from [ollama.com](https://ollama.com/).

2. **Pull the Llama 3 Model:**
   Open your terminal/command prompt and download the local model:
   ```bash
   ollama run llama3

### Installation

1. Clone the repository:
  git clone https://github.com/annBot2007/RAG-based-PDF-Chatbot.git
  cd RAG-based-PDF-Chatbot

2. Create and activate a virtual environment:
  python -m venv .venv
  # On Windows:
  .venv\Scripts\activate

3. Install these packages:
   pip install streamlit langchain-classic langchain-community langchain-ollama langchain-huggingface sentence-transformers faiss-cpu pypdf

4. Running the application:
   Ensure Ollama is running in the background
   streamlit run app.py
