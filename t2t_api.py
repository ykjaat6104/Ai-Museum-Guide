from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Enable CORS for frontend access
import os
import faiss
import json
import numpy as np
import asyncio
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import nest_asyncio
import re
import torch 
# Check for GPU availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# Apply nest_asyncio for async compatibility
nest_asyncio.apply()

app = Flask(__name__)
CORS(app)  # ✅ Allow requests from other origins (React frontend)

# Historical keywords - expanded list
history_keywords = {
    # General historical terms
    "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
    "dynasty", "emperor", "king", "queen", "ruler", "reign", "kingdom", "empire",
    "battle", "war", "treaty", "conquest", "rebellion", "revolution",
    "ancient", "medieval", "colonial", "pre-independence", "post-independence",
    
    # Indian historical figures
    "shivaji", "sambhaji", "maharana pratap", "chetak", "akbar", "ashoka", "buddha",
    "tipu sultan", "bahadur shah", "lakshmibai", "rani of jhansi", "tantia tope",
    "tanibai", "tara bai", "tarabai bhosale", "bajirao", "balaji vishwanath",
    "aurangzeb", "shah jahan", "babur", "humayun", "jahangir","hambirrao mohite",
    
    # Indian historical dynasties and empires
    "maratha", "mughal", "rajput", "peshwa", "gupta", "maurya", "chola",
    "vijayanagara", "chalukya", "pallava", "pandya", "satavahana", "kushana",
    "chhatrapati", "bhosale", "bhosle", "maratha empire", "delhi sultanate",
    
    # Indian historical events
    "battle of haldighati", "battle of panipat", "first war of independence", "sepoy mutiny",
    "struggle for independence", "salt march", "quit india", "partition",
    
    # Indian historical places
    "fort", "palace", "temple", "raigad", "sinhagad", "purandar", "agra", "delhi",
    "patliputra", "hampi", "golconda", "thanjavur", "mahabalipuram",
    
    # Indian historical artifacts
    "wagh nakh", "tiger claw", "sword", "peshwai", "seal", "coin", "inscription"
}

# Non-historical domains to filter out
off_domain_keywords = {
    # Technology related
    "artificial intelligence", "machine learning", "neural network", "deep learning", 
    "generative ai", "prompt engineering", "data science", "algorithm", "coding", "programming",
    "software", "hardware", "computer", "internet", "blockchain", "cryptocurrency",
    "chatgpt", "claude", "llm", "large language model", "autonomous", "robot",
    
    # Modern politics (post-2000)
    "election 2024", "recent election", "current government", "president biden", "prime minister modi",
    "trump", "kamala harris", "voting", "poll", "ballot", "democracy",
    
    # Entertainment
    "movie", "netflix", "amazon prime", "disney+", "film", "actor", "actress", "director",
    "music", "song", "album", "artist", "concert", "streaming", "video game",
    
    # Sports
    "cricket match", "football game", "world cup", "olympics", "tournament", "champion",
    "player", "team", "score", "ipl", "fifa", "nba", "tennis", "golf",
    
    # Finance & Business
    "stock market", "investment", "cryptocurrency", "bitcoin", "ethereum", "financial",
    "startup", "venture capital", "entrepreneur", "business model", "profit", "loss",
    
    # Science & Medicine (modern)
    "vaccine", "covid", "pandemic", "climate change", "global warming", "genetics",
    "quantum physics", "space exploration", "mars mission", "stem cell", "cloning"
}

@lru_cache(maxsize=1)
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def retrieve_relevant_text(query, k=3):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
        return None

    index = faiss.read_index(FAISS_INDEX_PATH)
    if index.ntotal == 0:
        return None

    with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
        text_map = json.load(f)

    embeddings_model = get_embeddings_model()
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    distances, retrieved_indices = index.search(query_embedding, k=k)
    relevant_contexts = [text_map[str(idx)] for idx in retrieved_indices[0] if str(idx) in text_map]
    return "\n\n".join(relevant_contexts) if relevant_contexts else None

def is_history_question(question):
    """Check if the question is related to history based on keywords"""
    question_lower = question.lower()
    
    # Check for off-domain keywords first
    for keyword in off_domain_keywords:
        if keyword.lower() in question_lower:
            return False
    
    # Check for history keywords
    return any(keyword.lower() in question_lower for keyword in history_keywords)

def check_question_relevance(question):
    """
    Determine if a question is within the domain expertise.
    
    Returns:
        tuple: (is_relevant, explanation)
    """
    question_lower = question.lower()
    
    # Check for explicitly off-domain topics
    for keyword in off_domain_keywords:
        if keyword.lower() in question_lower:
            return False, f"This question appears to be about {keyword}, which is outside my historical expertise. I specialize in Indian history, particularly the Maratha Empire and related historical topics."
    
    # Check if it contains history keywords
    has_history_keywords = any(keyword.lower() in question_lower for keyword in history_keywords)
    
    # If it has history keywords, it's relevant
    if has_history_keywords:
        return True, ""
    
    # For questions without clear indicators, perform additional checks
    
    # Check for time-related patterns (dates, centuries, eras)
    time_patterns = [
        r'\b\d{1,4}\s*(BC|BCE|AD|CE)\b',      # Years with era (400 BCE, 1600 CE)
        r'\b\d{1,2}(st|nd|rd|th)\s+century\b', # Centuries (19th century)
        r'\b(ancient|medieval|colonial|pre-independence|post-independence)\b' # Era terms
    ]
    
    for pattern in time_patterns:
        if re.search(pattern, question_lower):
            return True, ""
    
    # Questions about "who", "when", "what happened" are more likely to be historical
    historical_query_patterns = [
        r'\bwho\s+(was|were|ruled|conquered|founded|built|established)\b',
        r'\bwhen\s+(was|were|did)\b.+(built|founded|established|happen|occur)\b',
        r'\bwhat\s+happened\b',
        r'\bwhy\s+did\b.+(war|battle|conflict|rebellion)\b'
    ]
    
    for pattern in historical_query_patterns:
        if re.search(pattern, question_lower):
            return True, ""
    
    # If no clear indicators, lean towards rejecting
    return False, "I couldn't determine if your question is related to Indian history. I specialize in Indian historical topics, particularly around the Maratha Empire. Could you please rephrase with more historical context?"

def generate_history_prompt(question, relevant_context):
    return f"""
    You are an expert historian specializing in Indian history with deep knowledge of Indian History .
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

async def get_llm_response(prompt):
    llm = Ollama(model="mistral", timeout=120)
    return await llm.ainvoke(prompt)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        # Check question relevance first
        is_relevant, explanation = check_question_relevance(question)
        if not is_relevant:
            return jsonify({"error": explanation}), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(process_question(question))
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask/summarize", methods=["POST"])
def ask_and_summarize():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        # Check question relevance first
        is_relevant, explanation = check_question_relevance(question)
        if not is_relevant:
            return jsonify({"error": explanation}), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        full_response = loop.run_until_complete(process_question(question))
        prompt = f"""
        Summarize this historical text concisely:
        {full_response}
        Guidelines:
        1. Retain key facts and dates
        2. Maximum 3 sentences
        3. Maintain historical accuracy
        """
        summary = loop.run_until_complete(get_llm_response(prompt))
        return jsonify({"response": full_response, "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask/translate/<language>", methods=["POST"])
def ask_and_translate(language):
    data = request.get_json()
    question = data.get("question")

    if not question or not language:
        return jsonify({"error": "Question and language are required."}), 400

    # Validate language
    if language not in indian_languages.values():
        return jsonify({"error": f"Unsupported language: {language}. Supported languages are: {', '.join(indian_languages.values())}"}), 400

    try:
        # Check question relevance first
        is_relevant, explanation = check_question_relevance(question)
        if not is_relevant:
            return jsonify({"error": explanation}), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        full_response = loop.run_until_complete(process_question(question))
        translation_prompt = f"""
        Translate this historical text to {language} accurately:
        {full_response}
        Rules:
        1. Preserve all names and technical terms
        2. Maintain formal historical tone
        3. Keep dates and numbers unchanged
        """
        translated = loop.run_until_complete(get_llm_response(translation_prompt))
        return jsonify({"response": full_response, "translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "online"})

async def process_question(question):
    relevant_context = retrieve_relevant_text(question)
    prompt = generate_history_prompt(question, relevant_context)
    response = await get_llm_response(prompt)

    if "couldn't verify" not in response.lower() and "no information" not in response.lower():
        return response

    is_relevant, _ = check_question_relevance(question)
    if is_relevant:
        fallback_prompt = generate_history_prompt(question, None)
        return await get_llm_response(fallback_prompt)

    return "❌ This appears to be outside my historical expertise. I specialize in Indian history, particularly the Maratha Empire."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
