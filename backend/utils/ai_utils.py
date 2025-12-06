"""
Core AI utilities for Gemini integration and embeddings
"""
import google.generativeai as genai
from functools import lru_cache
import os

# Configure warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'

# Global AI state
gemini_model = None
api_key_configured = False

@lru_cache(maxsize=1)
def get_embeddings_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """Get cached embeddings model"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # Lazy import
        import torch
        # Clear any cached tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': False
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
                'show_progress_bar': False
            }
        )
        print(f" Embeddings model loaded successfully")
        return model
    except Exception as e:
        print(f" Embeddings model warning: {str(e)}")
        print(" Note: App will still work using online sources")
        return None

def setup_gemini(api_key, models_to_try=None):
    """
    Setup Gemini API with automatic model detection
    
    Args:
        api_key: Google Gemini API key
        models_to_try: List of model names to attempt (optional)
        
    Returns:
        bool: True if setup successful, False otherwise
    """
    global gemini_model, api_key_configured
    
    if not models_to_try:
        models_to_try = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'gemini-1.0-pro'
        ]
    
    try:
        genai.configure(api_key=api_key)
        
        # Try to get available models and find text generation models
        try:
            available_models = list(genai.list_models())
            print(f" Found {len(available_models)} available Gemini models")
            
            # Filter for models that support generateContent
            text_models = [
                m for m in available_models 
                if 'generateContent' in m.supported_generation_methods
            ]
            
            if text_models:
                # Use the first available text generation model
                model_name = text_models[0].name
                print(f" Trying model: {model_name}")
                
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Hello")
                
                if test_response and test_response.text:
                    print(f" Successfully configured Gemini model: {model_name}")
                    gemini_model = model
                    api_key_configured = True
                    return True
        except Exception as e:
            print(f" Model detection failed: {str(e)}, trying fallback...")
        
        # Fallback: Try predefined models
        for model_name in models_to_try:
            try:
                print(f"Trying fallback model: {model_name}")
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Hello")
                
                if test_response and test_response.text:
                    print(f" Successfully configured Gemini model: {model_name}")
                    gemini_model = model
                    api_key_configured = True
                    return True
            except Exception as e:
                print(f"{model_name} failed: {str(e)}")
                continue
        
        print("No compatible Gemini models found")
        return False
        
    except Exception as e:
        print(f"Gemini API Configuration Error: {str(e)}")
        return False

def get_gemini_model():
    """Get the configured Gemini model"""
    global gemini_model
    return gemini_model

def is_gemini_configured():
    """Check if Gemini is configured and ready"""
    global api_key_configured
    return api_key_configured

def generate_content(prompt, temperature=0.7, max_tokens=2048):
    """
    Generate content using Gemini
    
    Args:
        prompt: Text prompt
        temperature: Creativity level (0.0-1.0)
        max_tokens: Maximum response length
        
    Returns:
        str: Generated text or None if error
    """
    global gemini_model
    
    if not gemini_model:
        return None
    
    try:
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if response and response.text:
            return response.text
        return None
        
    except Exception as e:
        print(f"Gemini generation error: {str(e)}")
        return None

def generate_with_vision(prompt, image_data=None):
    """
    Generate content with vision capabilities (for artifact identification)
    
    Args:
        prompt: Text prompt
        image_data: PIL Image or image bytes
        
    Returns:
        str: Generated text or None if error
    """
    global gemini_model
    
    if not gemini_model or not image_data:
        return None
    
    try:
        if image_data:
            response = gemini_model.generate_content([prompt, image_data])
        else:
            response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            return response.text
        return None
        
    except Exception as e:
        print(f"Gemini vision error: {str(e)}")
        return None
