
import requests
from bs4 import BeautifulSoup
import re
import time
import json
import faiss
import numpy as np
from langdetect import detect
from googletrans import Translator
import os 

# List of URLs to scrape
urls = [
    'https://en.wikipedia.org/wiki/Tarabai',
    'https://en.wikipedia.org/wiki/Maharana_Pratap',
    'https://en.wikipedia.org/wiki/Chhatrapati_Sivaji_Maharaj',
    'https://hi.wikipedia.org/wiki/%E0%A4%B0%E0%A4%BE%E0%A4%A8%E0%A5%80_%E0%A4%B2%E0%A4%95%E0%A5%8D%E0%A4%B7%E0%A5%8D%E0%A4%AE%E0%A5%80%E0%A4%AC%E0%A4%BE%E0%A4%88',
    'https://en.wikipedia.org/wiki/Ashoka'
    
]

# File Paths
FAISS_INDEX_PATH = "faiss_index.bin"
TEXT_MAP_PATH = "faiss_text_map.json"

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove references or citations like [1]
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text.strip()

# Function to detect language
def detect_language(text):
    if len(text.strip()) < 5:  # Skip very short text
        return "unknown"
    try:
        return detect(text)
    except Exception as e:
        print(f"Error: {e}")
        return "unknown"  # Default to 'unknown' or 'en' for English

# Initialize the translator
translator = Translator()

# Function to translate text with retry logic
def translate_text_with_retry(text, target_language='en', retries=3):
    for attempt in range(retries):
        try:
            translated = translator.translate(text, dest=target_language)
            if translated.text is None:
                raise ValueError("Translation returned empty content")
            return translated.text
        except Exception as e:
            print(f"Error during translation (Attempt {attempt + 1}): {e}")
            time.sleep(2)  # Wait before retrying
    return text  # Return the original text after retries

# Function to generate dummy embeddings (you can replace this with a real model)
def generate_embeddings(text):
    return np.random.rand(768)  # Dummy embedding, replace with actual model

# Function to store data in FAISS
# Function to store data in FAISS
# Function to store data in FAISS
def store_in_faiss(data, index, embedding_dim=768):
    embeddings = generate_embeddings(data)  # Generate embeddings for the text
    
    # Ensure embeddings is a 2D array with shape (1, embedding_dim) and cast to float32
    embeddings = embeddings.reshape(1, -1).astype(np.float32)  # Reshape to (1, embedding_dim) and ensure dtype is float32
    
    # Check that the embedding dimension matches the FAISS index dimension
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match the index dimension {embedding_dim}")

    faiss.normalize_L2(embeddings)  # Normalize embeddings
    index.add(embeddings)  # Add to FAISS index

# Load the existing FAISS index or create a new one with the correct embedding size
def load_faiss_index(embedding_dim=768):
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        
        # Check if the existing FAISS index dimensionality matches the expected dimensionality
        if index.d != embedding_dim:
            print(f"Warning: FAISS index dimensionality {index.d} does not match the expected {embedding_dim}. Recreating the index...")
            # Recreate the FAISS index with the correct dimensionality
            index = faiss.IndexFlatL2(embedding_dim)  # Recreate with correct dimensionality
    else:
        index = faiss.IndexFlatL2(embedding_dim)  # Create a new index with the correct dimensionality

    return index
# Store data in JSON format
def store_data_in_json(data, filename=TEXT_MAP_PATH):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Load the existing FAISS index or create a new one

# Process multiple URLs
all_data = []  # List to store processed data from all URLs
index = load_faiss_index()  # Load the FAISS index

for url in urls:
    print(f"Scraping URL: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    
    # Clean and translate the paragraphs
    for p in paragraphs:
        cleaned_text = clean_text(p.get_text())  # Clean the text
        if cleaned_text:  # Skip empty paragraphs
            detected_language = detect_language(cleaned_text)  # Detect language
            print(f"Detected Language: {detected_language} - {cleaned_text[:100]}...")  # Print first 100 characters

            # Translate the text into Hindi and Marathi
            translated_hindi = translate_text_with_retry(cleaned_text, target_language='hi')
            translated_marathi = translate_text_with_retry(cleaned_text, target_language='mr')

            # Store data in JSON format
            data_entry = {
                "original": cleaned_text,
                "hindi": translated_hindi,
                "marathi": translated_marathi,
                "source": url
            }
            all_data.append(data_entry)

            # Store embeddings for original and translated text in FAISS
            store_in_faiss(cleaned_text, index)
            store_in_faiss(translated_hindi, index)
            store_in_faiss(translated_marathi, index)

# Save all processed data to JSON
store_data_in_json(all_data)

# Save the updated FAISS index
faiss.write_index(index, FAISS_INDEX_PATH)

print("Data stored in JSON and FAISS!")
