import os
import tempfile
import streamlit as st
from rag_engine import get_rag_chain, process_pdf

st.set_page_config(
    page_title="Local PDF RAG Chatbot", page_icon="📖", layout="centered"
)

st.title("Local & Free RAG Chatbot")
st.write(
    "Powered by Ollama (Llama 3) and Hugging Face. Zero API costs, 100%"
    " offline."
)
st.write(
    "Upload your files, get accurate responses summarized from your provided sources"
)

with st.sidebar:
  st.header("Upload Document")
  uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
  st.info("Make sure Ollama is running locally on your computer.")

if uploaded_file:
  if "vector_store" not in st.session_state:
    with st.spinner("Processing PDF locally (this may take a moment)..."):
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

      st.session_state.vector_store = process_pdf(tmp_path)
      st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store)
      os.unlink(tmp_path)
    st.success("Ready! Ask your questions below.")

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
      if "sources" in message:
        with st.expander("View Source Citations"):
          for i, doc in enumerate(message["sources"]):
            st.markdown(
                f"**Source {i+1} (Page"
                f" {doc.metadata.get('page', 'Unknown')})**"
            )
            st.text(doc.page_content[:300] + "...")

  if user_query := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("Thinking locally..."):
        response = st.session_state.rag_chain.invoke({"input": user_query})
        answer = response["answer"]
        sources = response["context"]

        st.markdown(answer)
        with st.expander("View Source Citations"):
          for i, doc in enumerate(sources):
            st.markdown(
                f"**Source {i+1} (Page"
                f" {doc.metadata.get('page', 'Unknown')})**"
            )
            st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
else:
  st.warning("Please upload a PDF document via the sidebar to start.")
