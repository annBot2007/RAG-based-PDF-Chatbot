import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  # Free local embeddings
from langchain_ollama import ChatOllama  # Free local LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(file_path: str):
  """Loads a PDF and uses a local HuggingFace model for embeddings."""
  loader = PyPDFLoader(file_path)
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200
  )
  chunks = text_splitter.split_documents(docs)

  # Free, local embedding model running on your CPU
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vector_store = FAISS.from_documents(chunks, embeddings)

  return vector_store


def get_rag_chain(vector_store):
  """Creates a RAG chain powered by local Ollama."""
  # Connects to your locally running Ollama instance
  llm = ChatOllama(model="llama3", temperature=0.2)

  system_prompt = (
      "You are an assistant for question-answering tasks. "
      "Use the following pieces of retrieved context to answer "
      "the question. If you don't know the answer, say that you "
      "don't know.\n\n"
      "{context}"
  )

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{input}"),
  ])

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  retriever = vector_store.as_retriever(
      search_type="similarity", search_kwargs={"k": 4}
  )
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  return rag_chain
