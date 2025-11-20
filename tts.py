import os
import faiss
import json
import numpy as np
import re
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import pyttsx3
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"  # Add your PDF folder path

# Supported Indian Languages for Translation
indian_languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

# Historical keywords to filter relevant questions
history_keywords = {
    # General historical terms
    "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit", "dynasty",
    "emperor", "king", "queen", "ruler", "reign", "kingdom", "empire", "battle", "war", "treaty",
    "conquest", "rebellion", "revolution", "ancient", "medieval", "colonial", "pre-independence",
    "post-independence",
   
    # Indian historical figures and dynasties
    "mauryan", "ashoka", "bindusara", "chandragupta", "gupta", "samudragupta", "vikramaditya",
    "kushan", "kanishka", "satavahana", "pallava", "chola", "rashtrakuta", "chalukya", "maratha",
    "shivaji", "sambhaji", "shahu", "peshwa", "bajirao", "balaji vishwanath", "balaji baji rao",
    "nana saheb", "raghunath rao", "madhav rao", "holkar", "scindia", "bhonsle", "gaikwad", "mughal",
    "babur", "humayun", "akbar", "jahangir", "shah jahan", "aurangzeb", "bahadur shah", "rajput",
    "prithviraj chauhan", "rana pratap", "rana sanga", "man singh", "amar singh", "sawai jai singh",
    "udai singh", "tipu sultan", "hyder ali", "maharana pratap", "chetak", "tanibai", "tara bai",
    "tarabai bhosale",
   
    # Indian historical places and monuments
    "fort", "palace", "temple", "raigad", "sinhagad", "purandar", "agra", "delhi", "patliputra",
    "hampi", "golconda", "thanjavur", "mahabalipuram", "ellora", "ajanta", "konark", "sanchi",
    "qutub minar", "gateway of india", "taj mahal", "red fort", "charminar", "mysore palace",
    "amer fort", "gwalior fort", "chittorgarh", "halebidu", "khajuraho", "elephanta caves",
    "jantar mantar", "brihadeeswarar temple", "ramappa temple", "rani ki vav", "indus valley",
    "harappa", "mohenjo daro",
   
    # Indian historical events and concepts
    "battle of panipat", "battle of haldighati", "battle of plassey", "battle of buxar",
    "first war of independence", "sepoy mutiny", "struggle for independence", "salt march",
    "quit india", "partition", "ashokan edicts",
   
    # Indian historical artifacts
    "wagh nakh", "tiger claw", "sword", "peshwai", "seal", "coin", "inscription", "script",
    "manuscript", "weapon", "armory", "shield", "turban", "armor"
}

# Non-historical domains to filter out
off_domain_keywords = {
    # Technology related
    "artificial intelligence", "machine learning", "neural network", "deep learning", "generative ai",
    "prompt engineering", "data science", "algorithm", "coding", "programming", "software", "hardware",
    "computer", "internet", "blockchain", "cryptocurrency", "chatgpt", "claude", "llm",
    "large language model", "autonomous", "robot", "gpu", "cpu",
   
    # Modern politics (post-2000)
    "election 2024", "recent election", "current government", "president biden", "prime minister modi",
    "trump", "kamala harris", "voting", "poll", "ballot", "democracy",
   
    # Entertainment
    "movie", "netflix", "amazon prime", "disney+", "film", "actor", "actress", "director", "music",
    "song", "album", "artist", "concert", "streaming", "video game",
   
    # Sports
    "cricket match", "football game", "world cup", "olympics", "tournament", "champion", "player",
    "team", "score", "ipl", "fifa", "nba", "tennis", "golf",
   
    # Finance & Business
    "stock market", "investment", "cryptocurrency", "bitcoin", "ethereum", "financial", "startup",
    "venture capital", "entrepreneur", "business model", "profit", "loss"
}

# Entertainment figures specifically (expanded)
entertainment_figures = {
    "shahrukh khan", "salman khan", "amitabh bachchan", "deepika padukone", "ranveer singh",
    "aamir khan", "priyanka chopra", "nick jonas", "ranbir kapoor", "alia bhatt", "kareena kapoor",
    "saif ali khan", "hrithik roshan", "ajay devgn", "kajol", "aishwarya rai", "shahid kapoor",
    "katrina kaif", "vicky kaushal", "anushka sharma", "virat kohli", "akshay kumar", "madhuri dixit",
    "sridevi", "rekha", "janhvi kapoor", "varun dhawan", "kiara advani", "sidharth malhotra",
    "rajkummar rao", "kangana ranaut", "taapsee pannu", "kartik aaryan", "bhumi pednekar",
    "ayushmann khurrana", "kriti sanon", "sara ali khan", "javed akhtar", "zoya akhtar", "farhan akhtar",
    "karan johar", "sanjay leela bhansali", "rohit shetty", "raj kapoor", "nargis", "dilip kumar",
    "waheeda rehman", "dharmendra", "hema malini", "sunny deol", "bobby deol", "govinda", "mithun chakraborty",
    "shah rukh", "srk", "sharukh", "big b", "bhaijaan", "bhai", "salmaan"
}

# Initialize multiple translator instances for reliability
def get_translator():
    for service_url in ['translate.google.com', 'translate.google.co.in']:
        try:
            return Translator(service_urls=[service_url])
        except:
            continue
   
    try:
        return Translator()
    except Exception as e:
        print(f"Failed to initialize translator: {e}")
        return None

# Create a reliable translation function
def translate_text(text, src='auto', dest='en'):
    """Translate text with fallback mechanisms"""
    if not text:
        return text
   
    for _ in range(3):
        translator = get_translator()
        if translator is None:
            time.sleep(1)
            continue
           
        try:
            result = translator.translate(text, src=src, dest=dest)
            if result and result.text:
                return result.text
        except Exception as e:
            print(f"Translation attempt failed: {e}")
            time.sleep(1)
   
    print("All translation attempts failed. Returning original text.")
    return text

# Language detection with fallback
def detect_language(text):
    """Detect language with pattern matching as backup"""
    if not text:
        return "en"
   
    for _ in range(2):
        translator = get_translator()
        if translator:
            try:
                detected = translator.detect(text)
                if detected and detected.lang:
                    print(f"Detected language: {detected.lang}")
                    if detected.lang in ['hi', 'mr', 'en']:
                        return detected.lang
            except Exception as e:
                print(f"Language detection error: {e}")
   
    # Fallback to pattern matching
    hindi_patterns = ["है", "का", "की", "में", "से", "कौन", "क्या", "थे", "और", "पर"]
    marathi_patterns = ["आहे", "काय", "कोण", "होते", "माझा", "तुमचा", "महाराज", "शिवाजी"]
   
    for pattern in hindi_patterns:
        if pattern in text:
            print("Pattern match detected: Hindi")
            return "hi"
   
    for pattern in marathi_patterns:
        if pattern in text:
            print("Pattern match detected: Marathi")
            return "mr"
   
    return "en"

# Load PDFs and create embeddings
def store_embeddings():
    pdf_texts = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                pdf_texts.extend(documents)
            except Exception as e:
                print(f"Error loading {file}: {e}")
   
    if not pdf_texts:
        print("No PDFs found or loaded")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(pdf_texts)
   
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
   
    embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
    with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(text_map, f, ensure_ascii=False, indent=4)
   
    print(f"Created FAISS index with {len(chunks)} chunks")
    return True

# Speech recognition function
def recognize_speech(language="en"):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8
    recognizer.dynamic_energy_threshold = True
   
    speech_lang = language
    if language == "mr":
        speech_lang = "hi-IN"
    elif language == "hi":
        speech_lang = "hi-IN"
    elif language == "en":
        speech_lang = "en-IN"
   
    with sr.Microphone() as source:
        print(f"Listening (language: {language})...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing speech...")
           
            try:
                query = recognizer.recognize_google(audio, language=speech_lang)
                print(f"User's Query: {query}")
                return query
            except sr.UnknownValueError:
                if speech_lang != "en-IN":
                    try:
                        print("Retrying with English...")
                        query = recognizer.recognize_google(audio, language="en-IN")
                        print(f"User's Query (recognized as English): {query}")
                        return query
                    except:
                        pass
                print("Sorry, I couldn't understand. Please repeat.")
                return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in speech recognition: {e}")
            return None

# Improved check if query is history-related
def is_history_related(query):
    query_lower = query.lower()
   
    # Check if any off-domain keyword is present (check this first)
    for keyword in off_domain_keywords:
        if keyword.lower() in query_lower:
            return False
           
    # Check for entertainment figures specifically (like actors)
    for figure in entertainment_figures:
        if figure in query_lower:
            return False
   
    # Check if any history keyword is present
    for keyword in history_keywords:
        if keyword.lower() in query_lower:
            return True
   
    # Default to allowing the query if no keywords match
    return True

# Retrieve relevant text from FAISS
def retrieve_relevant_text(query):
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
        print("FAISS index or text map not found")
        return None

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        if index.ntotal == 0:
            print("FAISS index is empty")
            return None

        with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
       
        query_embedding = np.array([embeddings_model.embed_query(query)], dtype=np.float32)

        if query_embedding.shape[1] != index.d:
            raise ValueError(f"Dimension mismatch: query {query_embedding.shape[1]}, index {index.d}")

        distances, retrieved_indices = index.search(query_embedding, k=3)
       
        relevant_contexts = []
        for i in range(min(3, len(retrieved_indices[0]))):
            idx = retrieved_indices[0][i]
            if idx != -1 and str(idx) in text_map:
                relevant_contexts.append(text_map[str(idx)])
       
        return "\n\n".join(relevant_contexts) if relevant_contexts else None

    except Exception as e:
        print(f"Error retrieving text: {e}")
        return None

# Generate historical prompt for LLM
def generate_history_prompt(question, relevant_context=None):
    prompt = f"""
    You are an expert historian specializing in Indian history with deep knowledge of the Maratha Empire.
    - Provide detailed, accurate responses only for historical topics
    - Include important dates, relationships, and historical significance
    - Structure your response with clear sections
   
    User Question:
    {question}
    """
   
    if relevant_context:
        prompt += f"""
        Relevant Context from Knowledge Base:
        {relevant_context}
        """
    else:
        prompt += """
        Note:
        No specific information found in the knowledge base for this query.
        Provide a general historical response based on your knowledge.
        """
   
    prompt += """
    Response Guidelines:
    1. Start with a brief introduction
    2. Provide key facts in bullet points
    3. Explain historical significance
    4. End with a conclusion
    5. If unsure, say "I couldn't verify this information"
    """
   
    return prompt

# Clean response text by removing markdown formatting
def clean_response_text(text):
    # Remove markdown-style formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold formatting
    text = re.sub(r'\*(.+?)\*', r'\1', text)      # Remove italic formatting
    text = re.sub(r'#+\s*', '', text)             # Remove heading markers
   
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
   
    return text.strip()

# Answer the question
def answer_question(query, lang="en"):
    original_query = query
    original_lang = lang
   
    print(f"Original query ({lang}): {query}")
   
    # Check if query is history-related
    if not is_history_related(query):
        not_in_domain_msg = "I'm sorry, but this question is not in my domain of Indian history. I can only answer questions about historical topics, figures, events, and artifacts related to Indian history."
       
        # Translate the message if needed
        if original_lang != "en":
            try:
                not_in_domain_msg = translate_text(not_in_domain_msg, src='en', dest=original_lang)
            except Exception as e:
                print(f"Translation error: {e}")
       
        return not_in_domain_msg
   
    # Translate non-English queries to English for processing
    if lang != "en":
        try:
            translated = translate_text(query, src=lang, dest='en')
            print(f"Translated query: {translated}")
            query = translated
        except Exception as e:
            print(f"Translation error: {e}")
   
    # Retrieve relevant context from FAISS
    relevant_context = retrieve_relevant_text(query)
   
    if relevant_context:
        print("Found relevant context in knowledge base")
    else:
        # Try alternative queries for better search results
        alt_queries = [
            f"{query} history",
            f"{query} Indian history",
            f"{query} biography"
        ]
       
        for alt_query in alt_queries:
            relevant_context = retrieve_relevant_text(alt_query)
            if relevant_context:
                print(f"Found context with alternative query: {alt_query}")
                break

    # Generate prompt for LLM
    prompt = generate_history_prompt(query, relevant_context)
   
    # Get response from LLM
    try:
        llm = Ollama(model="mistral")
        english_response = llm(prompt)
        english_response = clean_response_text(english_response)
    except Exception as e:
        print(f"LLM error: {e}")
        english_response = "I apologize, but I encountered an error processing your question. Please try again."
   
    print(f"\nEnglish Response:\n{english_response}")
   
    # Translate back to original language if needed
    if original_lang != "en":
        try:
            response = translate_text(english_response, src='en', dest=original_lang)
            print(f"Translated response to {original_lang}")
            return response
        except Exception as e:
            print(f"Translation error: {e}")
            return english_response
   
    return english_response

# Improved text-to-speech function that prioritizes Google TTS
def text_to_speech(text, lang='en'):
    if not text:
        return
       
    # Clean up previous audio file if exists
    if os.path.exists("response.mp3"):
        try:
            os.remove("response.mp3")
        except Exception as e:
            print(f"Error removing previous audio file: {e}")
   
    # Setup language code for gTTS
    tts_lang = lang
    if lang == "mr":  # Marathi - use Hindi as fallback if Marathi isn't fully supported
        tts_lang = "hi"
   
    # Always try Google TTS with better error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Generating speech with Google TTS (language: {tts_lang})...")
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.save("response.mp3")
           
            # Use pygame for more reliable playback
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load("response.mp3")
                pygame.mixer.music.play()
                print("Playing audio response with pygame...")
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
                return  # Success
            except ImportError:
                # Try playsound if pygame not available
                try:
                    from playsound import playsound
                    print("Playing audio response with playsound...")
                    playsound("response.mp3")
                    print("Audio playback complete.")
                    return  # Success
                except Exception as e:
                    print(f"Playsound error: {e}")
                    # Last resort - system default player
                    if os.name == 'nt':  # Windows
                        os.system("start response.mp3")
                    elif os.name == 'posix':  # Mac/Linux
                        os.system("xdg-open response.mp3 || open response.mp3")
                    # Wait based on text length
                    time.sleep(len(text) * 0.08)
                    return
                   
        except Exception as e:
            print(f"Google TTS attempt {attempt+1} failed: {e}")
            time.sleep(1)  # Wait before retry
   
    # Ultimate fallback - only if all Google TTS attempts fail
    print("All Google TTS attempts failed. Using pyttsx3 as last resort.")
    try:
        engine = pyttsx3.init()
        # Try to improve pyttsx3 voice quality
        voices = engine.getProperty('voices')
        if lang == 'hi' or lang == 'mr':
            # Try to find Hindi voice
            for voice in voices:
                if 'hindi' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        else:
            # For English, try to select a good quality voice
            for voice in voices:
                if 'david' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
       
        engine.setProperty('rate', 145)  # Slightly slower rate for clarity
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Backup TTS error: {e}")
        print("Unable to speak. Displaying text only.")

# Main function
def main():
    # Check if FAISS index exists, create if not
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Creating FAISS index...")
        if not store_embeddings():
            print("Failed to create FAISS index")
            return

    print("Indian History AI Guide activated. Say 'jack' to start.")
   
    # Current language for interaction
    current_lang = "en"
   
    # Initial activation
    while True:
        query = recognize_speech(current_lang)
       
        if not query:
            continue
           
        if any(word in query.lower() for word in ["jack", "jac", "jak", "hi jack"]):
            text_to_speech("Hello, I am your AI Guide. Please ask your historical question.", current_lang)
            break
        else:
            print("Please say 'jack' to activate the bot")
   
    # Main interaction loop
    while True:
        print(f"\nListening for your question in {current_lang}...")
       
        # Visual indicator that the system is ready for input
        print("=== Ready for your question (say 'exit' to quit) ===")
       
        query = recognize_speech(current_lang)
       
        if not query:
            continue
           
        if "exit" in query.lower() or "stop" in query.lower() or "quit" in query.lower():
            text_to_speech("Goodbye!", current_lang)
            break
           
        # Detect language of query
        detected_lang = detect_language(query)
        print(f"Detected language code: {detected_lang}")
       
        # Update current language for next interaction
        if detected_lang in ["en", "hi", "mr"]:
            current_lang = detected_lang
       
        # Check if query is history-related before processing
        if not is_history_related(query):
            print("Query is not history-related. Providing domain limitation message.")
            not_in_domain_msg = "I'm sorry, but this question is not in my domain of Indian history. I can only answer questions about historical topics, figures, events, and artifacts related to Indian history."
           
            # Translate the message if needed
            if current_lang != "en":
                try:
                    not_in_domain_msg = translate_text(not_in_domain_msg, src='en', dest=current_lang)
                except Exception as e:
                    print(f"Translation error: {e}")
           
            # Visual indicator that the system is speaking
            print("=== Speaking out-of-domain response ===")
            text_to_speech(not_in_domain_msg, current_lang)
            continue
       
        # Get response for history-related query
        print("Processing your question...")
        response = answer_question(query, detected_lang)
        print(f"\nAI Response ({detected_lang}):\n{response}")
       
        # Visual indicator that the system is speaking
        print("=== Speaking now (please wait) ===")
       
        # Speak response and wait for completion
        text_to_speech(response, detected_lang)
       
        # Visual indicator that speaking is done
        print("=== Speech completed ===")

if __name__ == "__main__":
    main()

