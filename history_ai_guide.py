import streamlit as st
import os
import faiss
import json
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from googletrans import Translator
from gtts import gTTS
import io
import pygame
import time
import threading
import queue

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# Initialize the Google Translate translator
translator = Translator()

# Supported Languages with wake words
LANGUAGES = {
    "english": {
        "wake_word": "hello guide",
        "code": "en",
        "voice": "en",
        "display": "English",
        "activation": "How can I help you with Indian history today?"
    },
    "hindi": {
        "wake_word": "namaste guide",
        "code": "hi",
        "voice": "hi",
        "display": "हिंदी",
        "activation": "मैं आपको भारतीय इतिहास के बारे में कैसे मदद कर सकता हूँ?"
    },
    "marathi": {
        "wake_word": "namaskar guide", 
        "code": "mr",
        "voice": "mr",
        "display": "मराठी",
        "activation": "मी तुम्हाला भारतीय इतिहासाबद्दल कशी मदत करू शकतो?"
    }
}

# ========== Core Functions ========== #
def load_pdfs_from_folder(folder_path=PDF_FOLDER):
    pdf_texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            pdf_texts.extend(documents)
    return pdf_texts

def store_embeddings():
    pdf_texts = load_pdfs_from_folder()
    if not pdf_texts:
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)
    if not chunks:
        return False

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)

    return True

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

    # Perform the search
    distances, retrieved_index = index.search(query_embedding, k=1)
    
    # Handle case where no results are found
    if retrieved_index[0][0] == -1 or str(retrieved_index[0][0]) not in text_map:
        return "No relevant historical information found."

    return text_map[str(retrieved_index[0][0])]


def check_keywords(question):
    history_keywords = {
        "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
        "shivaji", "sambhaji", "maharana pratap", "chetak", "battle of haldighati",
        "maratha", "mughal", "rajput", "akbar", "peshwa", "gupta", "ashoka", "buddha"
    }
    return any(keyword.lower() in question.lower() for keyword in history_keywords)

# ========== Language Processing ========== #
def generate_history_prompt(question):
    return f"""
    You are a historian specializing in Indian history.
    - Respond only if the topic is historical
    - Use verified sources (FAISS PDF data or approved history keywords)
    - If no historical context is found, reject the question

    USER QUESTION:
    {question}

    RESPONSE FORMAT:
    - Provide accurate historical information in English
    - If not historical, say: "This is not my expertise"
    - Keep response factual and concise
    """

def translate_response(text, target_lang):
    translated_text = translator.translate(text, dest=target_lang).text
    return translated_text

def answer_question(question):
    relevant_context = retrieve_relevant_text(question)
    
    if relevant_context or check_keywords(question):
        prompt = generate_history_prompt(question)
        llm = Ollama(model="mistral")
        english_response = llm(prompt)
        
        # Translate to target language if needed
        if current_language != "english":
            return translate_response(english_response, current_language)
        return english_response
    
    base_response = "This is not my expertise. I only provide historical knowledge about India."
    if current_language != "english":
        return translate_response(base_response, current_language)
    return base_response

# ========== Streamlit Interface ========== #
def speak_text(text, lang_code="en"):
    # Use gTTS directly for speech synthesis without pygame
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        mp3file = BytesIO()
        tts.write_to_fp(mp3file)
        mp3file.seek(0)

        st.audio(mp3file, format="audio/mp3")
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")

def main():
    st.title("Indian History AI Guide")

    # Check and create FAISS index if needed
    if not os.path.exists(FAISS_INDEX_PATH):
        st.write("\n⏳ Indexing historical texts...")
        if store_embeddings():
            st.write("✅ FAISS embeddings created successfully!")
        else:
            st.write("⚠️ Failed to load PDFs. Check the file path!")

    # User Input for Question
    question = st.text_input("Ask a question about Indian History:")

    if question:
        # Processing the question
        response = answer_question(question)

        # Displaying the response
        st.write(f"**Answer:** {response}")

        # Optional: Text-to-Speech output
        speak = st.checkbox("Enable Voice Response")
        if speak:
            lang_code = LANGUAGES[current_language]["voice"]
            speak_text(response, lang_code)

if __name__ == "__main__":
    main()
