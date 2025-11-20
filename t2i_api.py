from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
import os
import json
import asyncio
import base64
import requests
import numpy as np
import torch
from functools import lru_cache
import time
import faiss
import nest_asyncio
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Apply nest_asyncio for async compatibility
nest_asyncio.apply()


STABILITY_API_KEY = "Enter your Api key "
STABILITY_ENDPOINT = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# ---- FILE PATHS ----
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"

# Check GPU availability
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
print(f"Using device: {DEVICE} {'(GPU available)' if USE_GPU else '(CPU only)'}")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# ---- GUARDRAIL KEYWORDS ----
# Historical keywords for guardrail checks
history_keywords = {
    # General historical terms
    "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
    "dynasty", "emperor", "king", "queen", "ruler", "reign", "kingdom", "empire",
    "battle", "war", "treaty", "conquest", "rebellion", "revolution",
    "ancient", "medieval", "colonial", "pre-independence", "post-independence",
    
    # Indian historical figures and dynasties
    "mauryan", "ashoka", "bindusara", "chandragupta", "gupta", "samudragupta", "vikramaditya", 
    "kushan", "kanishka", "satavahana", "pallava", "chola", "rashtrakuta", "chalukya", 
    "maratha", "shivaji", "sambhaji", "shahu", "peshwa", "bajirao", "balaji vishwanath", 
    "balaji baji rao", "nana saheb", "raghunath rao", "madhav rao", "holkar", "scindia", 
    "bhonsle", "gaikwad", "mughal", "babur", "humayun", "akbar", "jahangir", "shah jahan", 
    "aurangzeb", "bahadur shah", "rajput", "prithviraj chauhan", "rana pratap", "rana sanga", 
    "man singh", "amar singh", "sawai jai singh", "udai singh", "tipu sultan", "hyder ali",
    "maharana pratap", "chetak", "tanibai", "tara bai", "tarabai bhosale",
    
    # Indian historical places and monuments
    "fort", "palace", "temple", "raigad", "sinhagad", "purandar", "agra", "delhi",
    "patliputra", "hampi", "golconda", "thanjavur", "mahabalipuram", "ellora", "ajanta", 
    "konark", "sanchi", "qutub minar", "gateway of india", "taj mahal", "red fort", 
    "charminar", "mysore palace", "amer fort", "gwalior fort", "chittorgarh", "halebidu", 
    "khajuraho", "elephanta caves", "jantar mantar", "brihadeeswarar temple", "ramappa temple", 
    "rani ki vav", "indus valley", "harappa", "mohenjo daro",
    
    # Indian historical events and concepts
    "battle of panipat", "battle of haldighati", "battle of plassey", "battle of buxar",
    "first war of independence", "sepoy mutiny", "struggle for independence", "salt march", 
    "quit india", "partition", "ashokan edicts",
    
    # Indian historical artifacts
    "wagh nakh", "tiger claw", "sword", "peshwai", "seal", "coin", "inscription",
    "script", "manuscript", "weapon", "armory", "shield", "turban", "armor"
}

# Non-historical domains to filter out
off_domain_keywords = {
    # Technology related
    "artificial intelligence", "machine learning", "neural network", "deep learning", 
    "generative ai", "prompt engineering", "data science", "algorithm", "coding", "programming",
    "software", "hardware", "computer", "internet", "blockchain", "cryptocurrency",
    "chatgpt", "claude", "llm", "large language model", "autonomous", "robot", "gpu", "cpu",
    
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
    "startup", "venture capital", "entrepreneur", "business model", "profit", "loss"
}

# ---- EMBEDDINGS MODEL ----
@lru_cache(maxsize=1)
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': DEVICE},  # Use GPU if available
        encode_kwargs={'normalize_embeddings': True}
    )

# Initialize GPU-based FAISS index if available
def get_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
        
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Move index to GPU if available (for faster search)
    if USE_GPU:
        res = faiss.StandardGpuResources()  # GPU resources
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU
        return gpu_index
    
    return index

# FAISS index and text map loading with caching
faiss_index = None
text_map = None

def load_resources():
    global faiss_index, text_map
    
    if faiss_index is None:
        faiss_index = get_faiss_index()
        
    if text_map is None and os.path.exists(TEXT_MAP_PATH):
        with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
            text_map = json.load(f)
            
    return faiss_index is not None and text_map is not None

# ---- RETRIEVE TEXT FROM FAISS ----
def retrieve_relevant_text(query, k=3):
    # Make sure resources are loaded
    if not load_resources() or faiss_index.ntotal == 0:
        return None

    start_time = time.time()
    
    # Get embeddings using GPU if available
    embeddings_model = get_embeddings_model()
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Perform search
    distances, retrieved_indices = faiss_index.search(query_embedding, k=k)
    relevant_contexts = [text_map[str(idx)] for idx in retrieved_indices[0] if str(idx) in text_map]
    
    end_time = time.time()
    print(f"Retrieval completed in {end_time - start_time:.4f} seconds")
    
    return "\n\n".join(relevant_contexts) if relevant_contexts else None

def check_question_relevance(question):
    """
    Determine if a question is within the historical domain expertise.
    
    Returns:
        tuple: (is_relevant, explanation)
    """
    question_lower = question.lower()
    
    # Check for explicitly off-domain topics
    for keyword in off_domain_keywords:
        if keyword.lower() in question_lower:
            return False, f"This question appears to be about {keyword}, which is outside my historical expertise. I specialize in Indian history and cannot generate images for non-historical topics."
    
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
    return False, "I couldn't determine if your question is related to Indian history. I only generate images for Indian historical topics. Please provide more historical context in your query."

# ---- CONSTRUCT PROMPT ----
def create_image_prompt(summary):
    """Create a more detailed prompt for image generation"""
    return f"Highly detailed digital painting of Indian historical scene. {summary.strip()}"

# ---- GENERATE SUMMARY ----
async def generate_history_summary(query):
    """Generate a historical summary suitable for image creation"""
    relevant_context = retrieve_relevant_text(query)
    
    prompt = f"""You are an expert historian specializing in Indian history.
    Create a vivid, visual description in 1-2 sentences for an image prompt based on this historical question.
    Focus on visual elements like scenery, clothing, architecture, artifacts, and people.

    Context from Knowledge Base:
    {relevant_context if relevant_context else "No direct references found in documents"}

    User Question: {query}

    Important instructions:
    1. Be historically accurate
    2. Include visual details (colors, settings, clothing, artifacts)
    3. Make it suitable for image generation
    4. Maximum 2 sentences
    5. Don't mention that this is for an image prompt
    """

    llm = Ollama(model="mistral", timeout=120)
    return await llm.ainvoke(prompt)

# ---- IMAGE GENERATION: STABILITY AI ----
def generate_image_with_stability(prompt_text):
    """Generate image using Stability AI API"""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}",
    }

    body = {
        "steps": 30,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 7,
        "samples": 1,
        "text_prompts": [
            {"text": prompt_text, "weight": 1},
            {"text": "blurry, distorted, bad anatomy, text, watermark, signature, modern elements, anachronism", "weight": -1}
        ]
    }

    response = requests.post(STABILITY_ENDPOINT, headers=headers, json=body)

    if response.status_code != 200:
        raise Exception(f"Image generation failed with status {response.status_code}: {response.text}")

    data = response.json()
    return data["artifacts"][0]["base64"]

@app.route("/api/generate", methods=["POST"])
def generate_historical_image():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing required field: question"}), 400
    
    question = data["question"]
    
    try:
        # Check if question is relevant to Indian history
        is_relevant, explanation = check_question_relevance(question)
        if not is_relevant:
            return jsonify({
                "error": "Domain error",
                "message": explanation,
                "success": False
            }), 400
        
        # Generate historical summary
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary = loop.run_until_complete(generate_history_summary(question))
        
        # Convert summary to image prompt
        image_prompt = create_image_prompt(summary)
        
        # Generate image
        image_base64 = generate_image_with_stability(image_prompt)
        
        return jsonify({
            "success": True,
            "summary": summary,
            "image_prompt": image_prompt,
            "image": image_base64
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": "Processing error",
            "message": str(e),
            "success": False
        }), 500

@app.route("/api/check", methods=["POST"])
def check_historical_relevance():
    """Check if a question is relevant to Indian history without generating an image"""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing required field: question"}), 400
    
    question = data["question"]
    is_relevant, explanation = check_question_relevance(question)
    
    return jsonify({
        "question": question,
        "is_relevant": is_relevant,
        "explanation": explanation if not is_relevant else "Question is relevant to Indian history"
    })

@app.route("/api/summary", methods=["POST"])
def generate_summary_only():
    """Generate historical summary without creating an image"""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing required field: question"}), 400
    
    question = data["question"]
    
    try:
        # Check if question is relevant
        is_relevant, explanation = check_question_relevance(question)
        if not is_relevant:
            return jsonify({
                "error": "Domain error",
                "message": explanation,
                "success": False
            }), 400
        
        # Generate historical summary
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summary = loop.run_until_complete(generate_history_summary(question))
        
        return jsonify({
            "success": True,
            "question": question,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({
            "error": "Processing error",
            "message": str(e),
            "success": False
        }), 500

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "gpu_available": USE_GPU,
        "device": DEVICE
    })

@app.route("/init", methods=["GET"])
def init_resources_endpoint():
    """Initialize resources endpoint - replaces @app.before_first_request"""
    print("Initializing resources...")
    success = load_resources()
    print("Resources loaded successfully" if success else "Resource loading failed")
    return jsonify({
        "status": "initialized" if success else "failed",
        "gpu_available": USE_GPU,
        "device": DEVICE
    })

# Initialize resources when the application starts
with app.app_context():
    print("Initializing resources on startup...")
    try:
        load_resources()
        print("Resources loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load resources on startup: {str(e)}")
        print("Resources will be loaded on first request")

if __name__ == '__main__':
    print("Starting Flask API service...")
    # For better GPU utilization, it's recommended to use a production WSGI server
    # like gunicorn or uwsgi instead of Flask's built-in server
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
