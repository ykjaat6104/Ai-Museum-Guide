
import os
import faiss
import json
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pygame import mixer
import threading
import queue
import time
from googletrans import Translator  # Using Google Translate for translation

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"
PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# Initialize pygame mixer
mixer.init()
mixer.set_num_channels(1)
voice = mixer.Channel(0)

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

# Global variables
current_language = "english"
sleeping = True
audio_queue = queue.Queue()
stop_event = threading.Event()  # Initialize the stop_event

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

    distances, retrieved_index = index.search(query_embedding, k=1)
    if retrieved_index[0][0] == -1 or str(retrieved_index[0][0]) not in text_map:
        return None

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
    # Translate using Google Translate API
    translated_text = translator.translate(text, dest=target_lang).text
    return translated_text

def answer_question(question):
    # First process in English
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

# ========== Speech Functions ========== #
def speak_text(text, lang_code="en"):
    try:
        # For Indian languages, first transliterate to improve TTS quality
        if lang_code in ["hi", "mr"]:
            # Convert text to Latin script for better TTS handling
            transliterated = translit_engine.translit_sentence(text, lang_code)
            print(f"Transliterated: {transliterated}")
            
            # Use transliterated text with English TTS
            tts = gTTS(text=transliterated, lang='en', slow=False)
        else:
            tts = gTTS(text=text, lang=lang_code, slow=False)
        
        mp3file = BytesIO()
        tts.write_to_fp(mp3file)
        mp3file.seek(0)
        
        sound = mixer.Sound(mp3file)
        voice.play(sound)
        
        while voice.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        # Fallback to English if native language fails
        if lang_code != "en":
            speak_text("Please wait while I respond in English", "en")

def speech_recognition_thread():
    global current_language, sleeping
    
    rec = sr.Recognizer()
    mic = sr.Microphone()
    rec.dynamic_energy_threshold = False
    rec.energy_threshold = 400
    
    while not stop_event.is_set():
        with mic as source:
            try:
                if sleeping:
                    print("\n🔇 Sleeping (say wake word to start)...")
                    rec.adjust_for_ambient_noise(source, duration=1)
                    audio = rec.listen(source, timeout=5, phrase_time_limit=5)
                    text = rec.recognize_google(audio, language="en").lower()
                    
                    # Check for any wake word
                    for lang, data in LANGUAGES.items():
                        if data["wake_word"].lower() in text:
                            current_language = lang
                            sleeping = False
                            print(f"\n🎤 Activated {data['display']} mode")
                            print(f"AI: {data['activation']}")
                            speak_text(data["activation"], data["voice"])
                            break
                
                else:
                    # Listen in current language
                    lang_data = LANGUAGES[current_language]
                    print(f"\n🎤 Listening in {lang_data['display']}...")
                    rec.adjust_for_ambient_noise(source, duration=1)
                    audio = rec.listen(source, timeout=8, phrase_time_limit=15)
                    
                    # For Indian languages, first try English recognition
                    text = rec.recognize_google(audio, language="en")
                    print(f"You: {text}")
                    
                    if "stop" in text.lower() or "exit" in text.lower():
                        sleeping = True
                        goodbye_msg = {
                            "english": "Goodbye!",
                            "hindi": "अलविदा!",
                            "marathi": "निरोप!"
                        }[current_language]
                        print(f"AI: {goodbye_msg}")
                        speak_text(goodbye_msg, lang_data["voice"])
                    else:
                        process_question(text)
                        
            except sr.UnknownValueError:
                if not sleeping:
                    print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                if "list" in str(e):
                    continue
                print(f"Error in speech recognition: {e}")

# ========== Main Processing ========== #
def process_question(question):
    if not question.strip():
        return
        
    print("\nProcessing your question...")
    response = answer_question(question)
    
    print(f"\nAI Response ({LANGUAGES[current_language]['display']}):")
    print(response)
    
    # Speak in selected language
    speak_text(response, LANGUAGES[current_language]["voice"])

def main():
    global stop_event
    
    print("\n" + "="*50)
    print("🏛️ Indian History AI Guide - Hands Free Mode")
    print("="*50)
    print("Wake words:")
    for lang, data in LANGUAGES.items():
        print(f"- {data['wake_word']} ({data['display']})")
    
    # Check and create FAISS index if needed
    if not os.path.exists(FAISS_INDEX_PATH):
        print("\n⏳ Indexing historical texts...")
        if store_embeddings():
            print("✅ FAISS embeddings created successfully!")
        else:
            print("⚠️ Failed to load PDFs. Check the file path!")
    
    # Start speech recognition thread
    speech_thread = threading.Thread(target=speech_recognition_thread)
    speech_thread.daemon = True
    speech_thread.start()
    
    try:
        while True:
            time.sleep(1)
            if stop_event.is_set():
                break
    
    except KeyboardInterrupt:
        print("\nExiting...")
        stop_event.set()
    finally:
        mixer.quit()

if __name__ == "__main__":
    main()





# import os
# import faiss
# import json
# import numpy as np
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
# import speech_recognition as sr
# from gtts import gTTS
# from io import BytesIO
# from pygame import mixer 
# import threading
# import queue
# import time
# # from ai4bharat.transliteration import XlitEngine  # Added for transliteration

# # File Paths
# FAISS_INDEX_PATH = "faiss_index.bin"
# TEXT_MAP_PATH = "faiss_text_map.json"
# PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# # Initialize pygame mixer
# mixer.init()
# mixer.set_num_channels(1)
# voice = mixer.Channel(0)

# # Initialize AI4Bharat transliteration engine
# # translit_engine = XlitEngine()

# # Supported Languages with wake words
# LANGUAGES = {
#     "english": {
#         "wake_word": "guide",
#         "code": "en",
#         "voice": "en",
#         "display": "English",
#         "activation": "How can I help you with Indian history today?"
#     },
#     "hindi": {
#         "wake_word": "hello guide",
#         "code": "hi",
#         "voice": "hi",
#         "display": "हिंदी",
#         "activation": "मैं आपको भारतीय इतिहास के बारे में कैसे मदद कर सकता हूँ?"
#     },
#     "marathi": {
#         "wake_word": "namaskar guide", 
#         "code": "mr",
#         "voice": "mr",
#         "display": "मराठी",
#         "activation": "मी तुम्हाला भारतीय इतिहासाबद्दल कशी मदत करू शकतो?"
#     }
# }

# # Strictly Historical Keywords
# history_keywords = {
#     "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
#     "shivaji", "sambhaji", "maharana pratap", "chetak", "battle of haldighati",
#     "maratha", "mughal", "rajput", "akbar", "peshwa", "gupta", "ashoka", "buddha"
# }

# # Global variables
# current_language = "english"
# sleeping = True
# audio_queue = queue.Queue()
# stop_event = threading.Event()

# # ========== Core Functions ========== #
# def load_pdfs_from_folder(folder_path=PDF_FOLDER):
#     pdf_texts = []
#     for file in os.listdir(folder_path):
#         if file.endswith(".pdf"):
#             pdf_path = os.path.join(folder_path, file)
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             pdf_texts.extend(documents)
#     return pdf_texts

# def store_embeddings():
#     pdf_texts = load_pdfs_from_folder()
#     if not pdf_texts:
#         return False

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#     chunks = text_splitter.split_documents(pdf_texts)
#     if not chunks:
#         return False

#     embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     faiss.write_index(index, FAISS_INDEX_PATH)

#     text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
#     with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
#         json.dump(text_map, f, ensure_ascii=False, indent=4)

#     return True

# def retrieve_relevant_text(query):
#     if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
#         return None

#     index = faiss.read_index(FAISS_INDEX_PATH)
#     if index.ntotal == 0:
#         return None

#     with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
#         text_map = json.load(f)

#     embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     query_embedding = np.array([embeddings_model.embed_query(query)], dtype=np.float32)

#     distances, retrieved_index = index.search(query_embedding, k=1)
#     if retrieved_index[0][0] == -1 or str(retrieved_index[0][0]) not in text_map:
#         return None

#     return text_map[str(retrieved_index[0][0])]

# def check_keywords(question):
#     return any(keyword.lower() in question.lower() for keyword in history_keywords)

# # ========== Language Processing ========== #
# def generate_history_prompt(question):
#     return f"""
#     You are a historian specializing in Indian history.
#     - Respond only if the topic is historical
#     - Use verified sources (FAISS PDF data or approved history keywords)
#     - If no historical context is found, reject the question

#     USER QUESTION:
#     {question}

#     RESPONSE FORMAT:
#     - Provide accurate historical information in English
#     - If not historical, say: "This is not my expertise"
#     - Keep response factual and concise
#     """

# def translate_response(text, target_lang):
#     if target_lang == "english":
#         return text
        
#     prompt = f"""
#     Translate this historical text to {LANGUAGES[target_lang]['display']}:
#     {text}
    
#     Rules:
#     - Preserve all historical facts
#     - Use formal language
#     - Maintain original meaning
#     - Output only the translation
#     """
    
#     llm = Ollama(model="mistral")
#     return llm(prompt)

# def answer_question(question):
#     # First process in English
#     relevant_context = retrieve_relevant_text(question)
    
#     if relevant_context or check_keywords(question):
#         prompt = generate_history_prompt(question)
#         llm = Ollama(model="mistral")
#         english_response = llm(prompt)
        
#         # Translate to target language if needed
#         if current_language != "english":
#             return translate_response(english_response, current_language)
#         return english_response
    
#     base_response = "This is not my expertise. I only provide historical knowledge about India."
#     if current_language != "english":
#         return translate_response(base_response, current_language)
#     return base_response

# # ========== Speech Functions ========== #
# def speak_text(text, lang_code="en"):
#     try:
#         # For Indian languages, first transliterate to improve TTS quality
#         if lang_code in ["hi", "mr"]:
#             # Convert text to Latin script for better TTS handling
#             transliterated = translit_engine.translit_sentence(text, lang_code)
#             print(f"Transliterated: {transliterated}")
            
#             # Use transliterated text with English TTS
#             tts = gTTS(text=transliterated, lang='en', slow=False)
#         else:
#             tts = gTTS(text=text, lang=lang_code, slow=False)
        
#         mp3file = BytesIO()
#         tts.write_to_fp(mp3file)
#         mp3file.seek(0)
        
#         sound = mixer.Sound(mp3file)
#         voice.play(sound)
        
#         while voice.get_busy():
#             time.sleep(0.1)
            
#     except Exception as e:
#         print(f"Error in text-to-speech: {e}")
#         # Fallback to English if native language fails
#         if lang_code != "en":
#             speak_text("Please wait while I respond in English", "en")

# def speech_recognition_thread():
#     global current_language, sleeping
    
#     rec = sr.Recognizer()
#     mic = sr.Microphone()
#     rec.dynamic_energy_threshold = False
#     rec.energy_threshold = 400
    
#     while not stop_event.is_set():
#         with mic as source:
#             try:
#                 if sleeping:
#                     print("\n🔇 Sleeping (say wake word to start)...")
#                     rec.adjust_for_ambient_noise(source, duration=1)
#                     audio = rec.listen(source, timeout=5, phrase_time_limit=5)
#                     text = rec.recognize_google(audio, language="en").lower()
                    
#                     # Check for any wake word
#                     for lang, data in LANGUAGES.items():
#                         if data["wake_word"].lower() in text:
#                             current_language = lang
#                             sleeping = False
#                             print(f"\n🎤 Activated {data['display']} mode")
#                             print(f"AI: {data['activation']}")
#                             speak_text(data["activation"], data["voice"])
#                             break
                
#                 else:
#                     # Listen in current language
#                     lang_data = LANGUAGES[current_language]
#                     print(f"\n🎤 Listening in {lang_data['display']}...")
#                     rec.adjust_for_ambient_noise(source, duration=1)
#                     audio = rec.listen(source, timeout=8, phrase_time_limit=15)
                    
#                     # For Indian languages, first try English recognition
#                     text = rec.recognize_google(audio, language="en")
#                     print(f"You: {text}")
                    
#                     if "stop" in text.lower() or "exit" in text.lower():
#                         sleeping = True
#                         goodbye_msg = {
#                             "english": "Goodbye!",
#                             "hindi": "अलविदा!",
#                             "marathi": "निरोप!"
#                         }[current_language]
#                         print(f"AI: {goodbye_msg}")
#                         speak_text(goodbye_msg, lang_data["voice"])
#                     else:
#                         process_question(text)
                        
#             except sr.UnknownValueError:
#                 if not sleeping:
#                     print("Could not understand audio")
#             except sr.RequestError as e:
#                 print(f"Could not request results; {e}")
#             except Exception as e:
#                 if "list" in str(e):
#                     continue
#                 print(f"Error in speech recognition: {e}")

# # ========== Main Processing ========== #
# def process_question(question):
#     if not question.strip():
#         return
        
#     print("\nProcessing your question...")
#     response = answer_question(question)
    
#     print(f"\nAI Response ({LANGUAGES[current_language]['display']}):")
#     print(response)
    
#     # Speak in selected language
#     speak_text(response, LANGUAGES[current_language]["voice"])

# def main():
#     global stop_event
    
#     print("\n" + "="*50)
#     print("🏛️ Indian History AI Guide - Hands Free Mode")
#     print("="*50)
#     print("Wake words:")
#     for lang, data in LANGUAGES.items():
#         print(f"- {data['wake_word']} ({data['display']})")
    
#     # Check and create FAISS index if needed
#     if not os.path.exists(FAISS_INDEX_PATH):
#         print("\n⏳ Indexing historical texts...")
#         if store_embeddings():
#             print("✅ FAISS embeddings created successfully!")
#         else:
#             print("⚠️ Failed to load PDFs. Check the file path!")
    
#     # Start speech recognition thread
#     speech_thread = threading.Thread(target=speech_recognition_thread)
#     speech_thread.daemon = True
#     speech_thread.start()
    
#     try:
#         while True:
#             time.sleep(1)
#             if stop_event.is_set():
#                 break
    
#     except KeyboardInterrupt:
#         print("\nExiting...")
#         stop_event.set()
#     finally:
#         mixer.quit()

# if __name__ == "__main__":
#     main()




# import os
# import faiss
# import json
# import numpy as np
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# import threading
# from gtts import gTTS
# from io import BytesIO
# from pygame import mixer
# import time
# from googletrans import Translator  # Using Google Translate for translation

# # File Paths
# FAISS_INDEX_PATH = "faiss_index.bin"
# TEXT_MAP_PATH = "faiss_text_map.json"
# PDF_FOLDER = r"E:\Ai Muesuem Guide\Books"

# # Initialize pygame mixer
# mixer.init()
# mixer.set_num_channels(1)
# voice = mixer.Channel(0)

# # Initialize the Google Translate translator
# translator = Translator()

# # Supported Languages with wake words
# LANGUAGES = {
#     "english": {
#         "wake_word": "hello guide",
#         "code": "en",
#         "voice": "en",
#         "display": "English",
#         "activation": "How can I help you with Indian history today?"
#     },
#     "hindi": {
#         "wake_word": "namaste guide",
#         "code": "hi",
#         "voice": "hi",
#         "display": "हिंदी",
#         "activation": "मैं आपको भारतीय इतिहास के बारे में कैसे मदद कर सकता हूँ?"
#     },
#     "marathi": {
#         "wake_word": "namaskar guide", 
#         "code": "mr",
#         "voice": "mr",
#         "display": "मराठी",
#         "activation": "मी तुम्हाला भारतीय इतिहासाबद्दल कशी मदत करू शकतो?"
#     }
# }

# # Global variables
# current_language = "english"
# sleeping = True
# audio_queue = queue.Queue()
# stop_event = threading.Event()  # Initialize the stop_event

# # ========== Core Functions ========== #
# def load_pdfs_from_folder(folder_path=PDF_FOLDER):
#     pdf_texts = []
#     for file in os.listdir(folder_path):
#         if file.endswith(".pdf"):
#             pdf_path = os.path.join(folder_path, file)
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             pdf_texts.extend(documents)
#     return pdf_texts

# def store_embeddings():
#     pdf_texts = load_pdfs_from_folder()
#     if not pdf_texts:
#         print("No PDF content found.")
#         return False

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
#     chunks = text_splitter.split_documents(pdf_texts)
#     print(f"Total Chunks: {len(chunks)}")  # Log the number of chunks

#     if not chunks:
#         return False

#     embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embeddings = np.array([embeddings_model.embed_query(chunk.page_content) for chunk in chunks], dtype=np.float32)

#     print(f"Embeddings Shape: {embeddings.shape}")  # Log the embeddings shape

#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     faiss.write_index(index, FAISS_INDEX_PATH)

#     text_map = {i: chunk.page_content for i, chunk in enumerate(chunks)}
#     with open(TEXT_MAP_PATH, "w", encoding="utf-8") as f:
#         json.dump(text_map, f, ensure_ascii=False, indent=4)
    
#     print("Embeddings stored successfully.")
#     return True

# def retrieve_relevant_text(query):
#     if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXT_MAP_PATH):
#         return None

#     index = faiss.read_index(FAISS_INDEX_PATH)
#     print(f"Index contains {index.ntotal} entries.")  # Log the number of entries

#     with open(TEXT_MAP_PATH, "r", encoding="utf-8") as f:
#         text_map = json.load(f)

#     embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     query_embedding = np.array([embeddings_model.embed_query(query)], dtype=np.float32)

#     print(f"Query Embedding Shape: {query_embedding.shape}")
#     print(f"Index Embedding Dimension: {index.d}")  # Log the index dimensions

#     distances, retrieved_index = index.search(query_embedding, k=1)
#     print(f"Retrieved Index: {retrieved_index}")  # Log the retrieved index

#     if retrieved_index[0][0] == -1 or str(retrieved_index[0][0]) not in text_map:
#         return None

#     return text_map[str(retrieved_index[0][0])]

# def check_keywords(question):
#     history_keywords = {
#         "museum", "monument", "artifact", "historical", "history", "heritage", "exhibit",
#         "shivaji", "sambhaji", "maharana pratap", "chetak", "battle of haldighati",
#         "maratha", "mughal", "rajput", "akbar", "peshwa", "gupta", "ashoka", "buddha"
#     }
#     matched_keywords = [keyword for keyword in history_keywords if keyword.lower() in question.lower()]
#     print(f"Matched Keywords: {matched_keywords}")  # Log the matched keywords
#     return len(matched_keywords) > 0

# # ========== Language Processing ========== #
# def generate_history_prompt(question):
#     prompt = f"""
#     You are a historian specializing in Indian history.
#     Respond to this question with accurate historical information:
#     {question}

#     Rules:
#     - Be factual and concise
#     - Use bullet points for clarity
#     - If not historical, say "This is not my expertise"
#     """
#     print(f"Generated Prompt: {prompt}")  # Log the generated prompt
#     return prompt

# def translate_response(text, target_lang):
#     # Translate using Google Translate API
#     translated_text = translator.translate(text, dest=target_lang).text
#     return translated_text

# def answer_question(question):
#     # First process in English
#     relevant_context = retrieve_relevant_text(question)
    
#     if relevant_context or check_keywords(question):
#         prompt = generate_history_prompt(question)
#         llm = Ollama(model="mistral")
#         english_response = llm(prompt)
        
#         # Translate to target language if needed
#         if current_language != "english":
#             return translate_response(english_response, current_language)
#         return english_response
    
#     base_response = "This is not my expertise. I only provide historical knowledge about India."
#     if current_language != "english":
#         return translate_response(base_response, current_language)
#     return base_response

# # ========== Speech Functions ========== #
# def speak_text(text, lang_code="en"):
#     try:
#         # For Indian languages, first transliterate to improve TTS quality
#         tts = gTTS(text=text, lang=lang_code, slow=False)
#         mp3file = BytesIO()
#         tts.write_to_fp(mp3file)
#         mp3file.seek(0)
        
#         sound = mixer.Sound(mp3file)
#         voice.play(sound)
        
#         while voice.get_busy():
#             time.sleep(0.1)
            
#     except Exception as e:
#         print(f"Error in text-to-speech: {e}")
#         # Fallback to English if native language fails
#         if lang_code != "en":
#             speak_text("Please wait while I respond in English", "en")

# # ========== Main Processing ========== #
# def process_question(question):
#     if not question.strip():
#         return
        
#     print("\nProcessing your question...")
#     response = answer_question(question)
    
#     print(f"\nAI Response ({LANGUAGES[current_language]['display']}):")
#     print(response)
    
#     # Speak in selected language
#     speak_text(response, LANGUAGES[current_language]["voice"])

# def main():
#     print("\n" + "="*50)
#     print("🏛️ Indian History AI Guide - Hands Free Mode")
#     print("="*50)
#     print("Wake words:")
#     for lang, data in LANGUAGES.items():
#         print(f"- {data['wake_word']} ({data['display']})")
    
#     # Check and create FAISS index if needed
#     if not os.path.exists(FAISS_INDEX_PATH):
#         print("\n⏳ Indexing historical texts...")
#         if store_embeddings():
#             print("✅ FAISS embeddings created successfully!")
#         else:
#             print("⚠️ Failed to load PDFs. Check the file path!")
    
#     # Start speech recognition thread
#     # speech_thread = threading.Thread(target=speech_recognition_thread)  # Uncomment if needed
#     # speech_thread.daemon = True
#     # speech_thread.start()
    
#     try:
#         while True:
#             time.sleep(1)
    
#     except KeyboardInterrupt:
#         print("\nExiting...")
#     finally:
#         mixer.quit()

# if __name__ == "__main__":
#     main()
