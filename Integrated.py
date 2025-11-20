import os
import faiss
import json
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.llms import Ollama
import asyncio
import nest_asyncio
from functools import lru_cache

# Configure environment to prevent symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'

# Apply nest_asyncio to fix event loop issues
nest_asyncio.apply()

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"

# Supported Indian Languages for Translation
indian_languages = {
    "English": "English",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Telugu": "Telugu",
    "Marathi": "Marathi",
    "Tamil": "Tamil",
    "Urdu": "Urdu",
    "Gujarati": "Gujarati",
    "Malayalam": "Malayalam",
    "Kannada": "Kannada",
    "Odia": "Odia",
    "Punjabi": "Punjabi",
    "Assamese": "Assamese"
}

# Enhanced Historical Keywords with aliases
history_keywords = {
    "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
    # "shivaji", "sambhaji", "maharana pratap", "chetak", "battle of haldighati",
    # "maratha", "mughal", "rajput", "akbar", "peshwa", "gupta", "ashoka", "buddha",
    # "tanibai", "tara bai", "tarabai bhosale", "wagh nakh", "tiger claw",
    # "chhatrapati", "bhosale", "bhosle", "maratha empire"
}

# Cache embeddings model to prevent reinitialization
@lru_cache(maxsize=1)
def get_embeddings_model():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Failed to load embeddings model: {str(e)}")
        return None

def retrieve_relevant_text(query, k=3):
    try:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
            st.warning("Knowledge base not found. Please upload documents first.")
            return None

        index = faiss.read_index(FAISS_INDEX_PATH)
        if index.ntotal == 0:
            return None

        with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        embeddings_model = get_embeddings_model()
        if not embeddings_model:
            return None

        query_embedding = embeddings_model.embed_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        if query_embedding.shape[1] != index.d:
            st.error(f"Dimension mismatch detected. Expected {index.d} but got {query_embedding.shape[1]}")
            return None

        distances, retrieved_indices = index.search(query_embedding, k=k)
        
        relevant_contexts = []
        for i in range(k):
            idx = retrieved_indices[0][i]
            if idx != -1 and str(idx) in text_map:
                relevant_contexts.append(text_map[str(idx)])
        
        return "\n\n".join(relevant_contexts) if relevant_contexts else None
    
    except Exception as e:
        st.error(f"Error retrieving text: {str(e)}")
        return None

def check_keywords(question):
    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in history_keywords)

def generate_history_prompt(question, relevant_context):
    return f"""
    You are an expert historian specializing in Indian history with deep knowledge of Maratha Empire.
    - Provide detailed, accurate responses only for historical topics
    - Include important dates, relationships, and historical significance
    - Structure your response with clear sections
    
    **Relevant Context from Knowledge Base:**
    {relevant_context if relevant_context else "No direct references found in documents"}
    
    **User Question:**
    {question}
    
    **Response Guidelines:**
    1. Start with a brief introduction
    2. Provide key facts in bullet points
    3. Explain historical significance
    4. End with sources/context used
    5. If unsure, say "I couldn't verify this information"
    """

async def answer_question(question):
    try:
        relevant_context = retrieve_relevant_text(question)
        
        if relevant_context:
            prompt = generate_history_prompt(question, relevant_context)
            llm = Ollama(model="mistral", timeout=120)
            response = await llm.ainvoke(prompt)
            
            if "couldn't verify" not in response.lower() and "no information" not in response.lower():
                return response
        
        if check_keywords(question):
            prompt = generate_history_prompt(question, relevant_context=None)
            llm = Ollama(model="mistral", timeout=120)
            return await llm.ainvoke(prompt)
            
        return "❌ This appears to be outside my historical expertise."
    
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

async def translate_response(response_text, selected_language):
    if selected_language == "English":
        return response_text

    try:
        prompt = f"""
        Translate this historical text to {selected_language} accurately:
        {response_text}
        Rules:
        1. Preserve all names and technical terms
        2. Maintain formal historical tone
        3. Keep dates and numbers unchanged
        """
        
        llm = Ollama(model="mistral", timeout=120)
        return await llm.ainvoke(prompt)
    except Exception as e:
        return f"Translation error: {str(e)}"

async def summarize_response(response_text):
    try:
        prompt = f"""
        Summarize this historical text concisely:
        {response_text}
        Guidelines:
        1. Retain key facts and dates
        2. Maximum 3 sentences
        3. Maintain historical accuracy
        """
        
        llm = Ollama(model="mistral", timeout=120)
        return await llm.ainvoke(prompt)
    except Exception as e:
        return f"Summarization error: {str(e)}"

def main():
    st.set_page_config(
        page_title="🏛️ Indian History AI Guide",
        page_icon="📜",
        layout="wide"
    )
    
    st.title("🏛️ Indian History AI Guide")
    st.subheader("📜 Ask about historical figures, battles, monuments & more!")

    user_query = st.text_input("❓ Ask a history-related question:")

    if st.button("Get Answer"):
        if user_query.strip():
            with st.spinner("Consulting historical sources..."):
                try:
                    response = asyncio.run(answer_question(user_query))
                    
                    if response.startswith("❌"):
                        st.error(response)
                    else:
                        st.session_state["latest_response"] = response
                        st.subheader("📜 Answer:")
                        st.markdown(response)
                except Exception as e:
                    st.error(f"Failed to get answer: {str(e)}")
        else:
            st.warning("⚠️ Please enter a valid question.")

    if "latest_response" in st.session_state:
        response_text = st.session_state["latest_response"]

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Summarize Response"):
                with st.spinner("Creating summary..."):
                    try:
                        summarized_text = asyncio.run(summarize_response(response_text))
                        st.subheader("📝 Summarized Text:")
                        st.write(summarized_text)
                    except Exception as e:
                        st.error(f"Failed to summarize: {str(e)}")

        with col2:
            selected_language = st.selectbox(
                "🌍 Translate Response To:", 
                list(indian_languages.keys()), 
                key="lang_select"
            )
            if st.button("Translate Response"):
                with st.spinner(f"Translating to {selected_language}..."):
                    try:
                        translated_text = asyncio.run(translate_response(response_text, selected_language))
                        st.subheader(f"🌍 Translated Text ({selected_language}):")
                        st.write(translated_text)
                    except Exception as e:
                        st.error(f"Failed to translate: {str(e)}")

    st.markdown("---")
    st.markdown("📌 **Note:** This AI answers only history-related queries. Off-topic questions will be rejected.")

if __name__ == "__main__":
    # Clear caches on startup
    get_embeddings_model.cache_clear()
    main() 

# import os
# import json
# import faiss
# import numpy as np
# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
# import torch
# import nest_asyncio
# from functools import lru_cache

# # Configure environment to prevent symlinks warning
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'

# # Apply nest_asyncio to fix event loop issues
# nest_asyncio.apply()

# # File Paths
# FAISS_INDEX_PATH = "faiss_index.bin"
# TEXT_MAP_PATH = "faiss_text_map.json"

# # Supported Indian Languages for Translation
# indian_languages = {
#     "English": "English",
#     "Hindi": "Hindi",
#     "Bengali": "Bengali",
#     "Telugu": "Telugu",
#     "Marathi": "Marathi",
#     "Tamil": "Tamil",
#     "Urdu": "Urdu",
#     "Gujarati": "Gujarati",
#     "Malayalam": "Malayalam",
#     "Kannada": "Kannada",
#     "Odia": "Odia",
#     "Punjabi": "Punjabi",
#     "Assamese": "Assamese"
# }

# # Enhanced Historical Keywords with aliases
# history_keywords = {
#     "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
#     "shivaji", "sambhaji", "maharana pratap", "chetak", "battle of haldighati",
#     "maratha", "mughal", "rajput", "akbar", "peshwa", "gupta", "ashoka", "buddha",
#     "tanibai", "tara bai", "tarabai bhosale", "wagh nakh", "tiger claw",
#     "chhatrapati", "bhosale", "maratha empire"
# }

# # Cache embeddings model to prevent reinitialization
# @lru_cache(maxsize=1)
# def get_embeddings_model():
#     try:
#         return HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2",
#             model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )
#     except Exception as e:
#         st.error(f"Failed to load embeddings model: {str(e)}")
#         return None

# # FAISS retrieval function with GPU support
# def retrieve_relevant_text(query, k=3):
#     try:
#         if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
#             st.warning("Knowledge base not found. Please upload documents first.")
#             return None

#         # Load FAISS index (GPU or CPU)
#         index = faiss.read_index(FAISS_INDEX_PATH)
#         if torch.cuda.is_available():
#             res = faiss.StandardGpuResources()
#             index = faiss.index_cpu_to_gpu(res, 0, index)

#         if index.ntotal == 0:
#             return None

#         with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
#             text_map = json.load(f)

#         embeddings_model = get_embeddings_model()
#         if not embeddings_model:
#             return None

#         query_embedding = embeddings_model.embed_query(query)
#         query_embedding = np.array([query_embedding], dtype=np.float32)

#         if query_embedding.shape[1] != index.d:
#             st.error(f"Dimension mismatch detected. Expected {index.d} but got {query_embedding.shape[1]}")
#             return None

#         distances, retrieved_indices = index.search(query_embedding, k=k)

#         relevant_contexts = []
#         for i in range(k):
#             idx = retrieved_indices[0][i]
#             if idx != -1 and str(idx) in text_map:
#                 relevant_contexts.append(text_map[str(idx)])

#         return "\n\n".join(relevant_contexts) if relevant_contexts else None
    
#     except Exception as e:
#         st.error(f"Error retrieving text: {str(e)}")
#         return None

# # Check if the query is related to history
# def check_keywords(question):
#     question_lower = question.lower()
#     return any(keyword.lower() in question_lower for keyword in history_keywords)

# # Generate historical prompt for LLM
# def generate_history_prompt(question, relevant_context):
#     return f"""
#     You are an expert historian specializing in Indian history with deep knowledge of Maratha Empire.
#     - Provide detailed, accurate responses only for historical topics
#     - Include important dates, relationships, and historical significance
#     - Structure your response with clear sections
    
#     **Relevant Context from Knowledge Base:**
#     {relevant_context if relevant_context else "No direct references found in documents"}
    
#     **User Question:**
#     {question}
    
#     **Response Guidelines:**
#     1. Start with a brief introduction
#     2. Provide key facts in bullet points
#     3. Explain historical significance
#     4. End with sources/context used
#     5. If unsure, say "I couldn't verify this information"
#     """

# # Answer the question with GPU acceleration and detailed context
# async def answer_question(question):
#     try:
#         relevant_context = retrieve_relevant_text(question)

#         if relevant_context:
#             prompt = generate_history_prompt(question, relevant_context)
#             llm = Ollama(model="mistral", timeout=120)
#             response = await llm.ainvoke(prompt)

#             if "couldn't verify" not in response.lower() and "no information" not in response.lower():
#                 return response

#         if check_keywords(question):
#             prompt = generate_history_prompt(question, relevant_context=None)
#             llm = Ollama(model="mistral", timeout=120)
#             return await llm.ainvoke(prompt)

#         return "❌ This appears to be outside my historical expertise."

#     except Exception as e:
#         return f"❌ An error occurred: {str(e)}"

# async def translate_response(response_text, selected_language):
#     if selected_language == "English":
#         return response_text

#     try:
#         prompt = f"""
#         Translate this historical text to {selected_language} accurately:
#         {response_text}
#         Rules:
#         1. Preserve all names and technical terms
#         2. Maintain formal historical tone
#         3. Keep dates and numbers unchanged
#         """
        
#         llm = Ollama(model="mistral", timeout=120)
#         return await llm.ainvoke(prompt)
#     except Exception as e:
#         return f"Translation error: {str(e)}"

# async def summarize_response(response_text):
#     try:
#         prompt = f"""
#         Summarize this historical text concisely:
#         {response_text}
#         Guidelines:
#         1. Retain key facts and dates
#         2. Maximum 3 sentences
#         3. Maintain historical accuracy
#         """
        
#         llm = Ollama(model="mistral", timeout=120)
#         return await llm.ainvoke(prompt)
#     except Exception as e:
#         return f"Summarization error: {str(e)}"

# # Streamlit UI for displaying the answer and responses
# def main():
#     st.set_page_config(
#         page_title="🏛️ Indian History AI Guide",
#         page_icon="📜",
#         layout="wide"
#     )
    
#     st.title("🏛️ Indian History AI Guide")
#     st.subheader("📜 Ask about historical figures, battles, monuments & more!")

#     user_query = st.text_input("❓ Ask a history-related question:")

#     if st.button("Get Answer"):
#         if user_query.strip():
#             with st.spinner("Consulting historical sources..."):
#                 try:
#                     response = asyncio.run(answer_question(user_query))
                    
#                     if response.startswith("❌"):
#                         st.error(response)
#                     else:
#                         st.session_state["latest_response"] = response
#                         st.subheader("📜 Answer:")
#                         st.markdown(response)
#                 except Exception as e:
#                     st.error(f"Failed to get answer: {str(e)}")
#         else:
#             st.warning("⚠️ Please enter a valid question.")

#     if "latest_response" in st.session_state:
#         response_text = st.session_state["latest_response"]

#         col1, col2 = st.columns(2)

#         with col1:
#             if st.button("Summarize Response"):
#                 with st.spinner("Creating summary..."):
#                     try:
#                         summarized_text = asyncio.run(summarize_response(response_text))
#                         st.subheader("📝 Summarized Text:")
#                         st.write(summarized_text)
#                     except Exception as e:
#                         st.error(f"Failed to summarize: {str(e)}")

#         with col2:
#             selected_language = st.selectbox(
#                 "🌍 Translate Response To:", 
#                 list(indian_languages.keys()), 
#                 key="lang_select"
#             )
#             if st.button("Translate Response"):
#                 with st.spinner(f"Translating to {selected_language}..."):
#                     try:
#                         translated_text = asyncio.run(translate_response(response_text, selected_language))
#                         st.subheader(f"🌍 Translated Text ({selected_language}):")
#                         st.write(translated_text)
#                     except Exception as e:
#                         st.error(f"Failed to translate: {str(e)}")

#     st.markdown("---")
#     st.markdown("📌 **Note:** This AI answers only history-related queries. Off-topic questions will be rejected.")

# if __name__ == "__main__":
#     # Clear caches on startup
#     get_embeddings_model.cache_clear()
#     main()
