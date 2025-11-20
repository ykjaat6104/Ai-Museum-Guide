import os
import faiss
import json
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_METADATA_PATH = "processed_pdfs.json"
PDF_FOLDER = r"E:\Ai Museum Guide\Books" 

# ✅ Function to Track Processed PDFs
def load_processed_pdfs():
    if os.path.exists(PDF_METADATA_PATH):
        with open(PDF_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_processed_pdfs(metadata):
    with open(PDF_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

# ✅ Load PDFs & Extract New Data Only
def load_pdfs_from_folder(folder_path=PDF_FOLDER):
    processed_pdfs = load_processed_pdfs()
    pdf_texts = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf") and file not in processed_pdfs:
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            pdf_texts.extend(documents)
            processed_pdfs[file] = True  # Mark as processed

    save_processed_pdfs(processed_pdfs)
    return pdf_texts

# ✅ Incrementally Store Embeddings in FAISS
def update_embeddings():
    pdf_texts = load_pdfs_from_folder()
    if not pdf_texts:
        return False  # No new PDFs to process

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)

    if not chunks:
        return False

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

    # Load existing FAISS index or create new one
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)  # Append new embeddings
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Update text map
    if os.path.exists(TEXT_MAP_PATH):
        with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
            text_map = json.load(f)
    else:
        text_map = {}

    text_map.update({len(text_map) + i: chunk.page_content for i, chunk in enumerate(chunks)})

    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)

    return True

# ✅ Retrieve Relevant Context from FAISS
def retrieve_relevant_text(query):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
        return None

    index = faiss.read_index(FAISS_INDEX_PATH)
    if index.ntotal == 0:
        return None

    with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
        text_map = json.load(f)

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = np.array([embeddings_model.embed_query(query)], dtype=np.float32)

    distances, retrieved_index = index.search(query_embedding, k=1)

    if retrieved_index[0][0] == -1 or str(retrieved_index[0][0]) not in text_map:
        return None

    return text_map[str(retrieved_index[0][0])]

# ✅ Generate Historical Prompt
def generate_history_prompt(question, relevant_context):
    return f"""
    You are a historian specializing in Indian history.
    - Answer **only** if the topic is historical.
    - Use **verified sources** (FAISS data or history-related keywords).
    - If no historical context is found, reject the question.

    **HISTORICAL CONTEXT (if available):**
    {relevant_context if relevant_context else "⚠️ No direct reference found."}

    **USER QUESTION:**
    {question}

    **YOUR RESPONSE:**
    - If relevant history is found, provide an accurate answer.
    - Otherwise, say: **"❌ This is not my expertise. I only provide historical knowledge."**
    """

# ✅ Answer Question using FAISS, Keywords, or Reject
def answer_question(question):
    relevant_context = retrieve_relevant_text(question)
    if relevant_context:
        prompt = generate_history_prompt(question, relevant_context)
        llm = Ollama(model="mistral")
        return llm(prompt)

    return "❌ This is not my expertise."

# ✅ Streamlit UI with PDF Upload
def main():
    st.set_page_config(page_title="🏛️ Indian History Guide", page_icon="📜", layout="wide")
    st.title("🏛️ Indian History AI Guide")
    st.subheader("📜 Ask about historical figures, battles, monuments & more!")

    uploaded_files = st.file_uploader("📂 Upload PDF files to expand knowledge:", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        with st.spinner("🔍 Updating knowledge base..."):
            if update_embeddings():
                st.success("✅ PDFs processed successfully!")
            else:
                st.warning("⚠️ No new information added.")

    user_query = st.text_input("❓ Ask a history-related question:")

    if st.button("Get Answer"):
        if user_query.strip():
            response = answer_question(user_query)
            if response.startswith("❌"):
                st.error(response)
            else:
                st.session_state["latest_response"] = response
                st.subheader("📜 Answer:")
                st.write(response)
        else:
            st.warning("⚠️ Please enter a valid question.")

    if "latest_response" in st.session_state:
        response_text = st.session_state["latest_response"]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Summarize Response"):
                summarized_text = answer_question(f"Summarize: {response_text}")
                st.subheader("📝 Summarized Text:")
                st.write(summarized_text)

        with col2:
            selected_language = st.selectbox("🌍 Translate Response To:", list(indian_languages.keys()), key="lang_select")
            if st.button("Translate Response"):
                translated_text = answer_question(f"Translate to {selected_language}: {response_text}")
                st.subheader(f"🌍 Translated Text ({selected_language}):")
                st.write(translated_text)

if __name__ == "__main__":
    main()
