"""
Translation routes for multilingual support
"""
from flask import Blueprint, request, jsonify
from datetime import datetime

translate_bp = Blueprint('translate', __name__)

# Supported languages
SUPPORTED_LANGUAGES = {
    'english': 'English',
    'spanish': 'Spanish (Español)',
    'french': 'French (Français)',
    'german': 'German (Deutsch)',
    'italian': 'Italian (Italiano)',
    'portuguese': 'Portuguese (Português)',
    'russian': 'Russian (Русский)',
    'chinese': 'Chinese (中文)',
    'japanese': 'Japanese (日本語)',
    'korean': 'Korean (한국어)',
    'arabic': 'Arabic (العربية)',
    'hindi': 'Hindi (हिंदी)',
    'bengali': 'Bengali (বাংলা)',
    'urdu': 'Urdu (اردو)',
    'tamil': 'Tamil (தமிழ்)',
    'telugu': 'Telugu (తెలుగు)',
    'marathi': 'Marathi (मराठी)',
    'gujarati': 'Gujarati (ગુજરાતી)',
    'kannada': 'Kannada (ಕನ್ನಡ)',
    'malayalam': 'Malayalam (മലയാളം)'
}

@translate_bp.route('/translate', methods=['POST'])
def translate_text():
    """
    Translate text to different languages
    
    Expected JSON:
        {
            "text": "Historical content to translate...",
            "language": "hindi"
        }
        
    Returns:
        {
            "original_text": "...",
            "translated_text": "...",
            "target_language": "Hindi",
            "timestamp": "..."
        }
    """
    from utils.ai_utils import is_gemini_configured, generate_content
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        target_language = data.get('language', '').strip().lower()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not target_language:
            return jsonify({'error': 'Target language is required'}), 400
        
        if target_language not in SUPPORTED_LANGUAGES:
            return jsonify({
                'error': f'Unsupported language: {target_language}',
                'supported_languages': list(SUPPORTED_LANGUAGES.keys())
            }), 400
        
        # No translation needed for English
        if target_language == 'english':
            return jsonify({
                'original_text': text,
                'translated_text': text,
                'target_language': SUPPORTED_LANGUAGES[target_language],
                'timestamp': datetime.now().isoformat()
            })
        
        # Check if AI is configured
        if not is_gemini_configured():
            return jsonify({
                'error': 'AI translation requires API configuration',
                'suggestion': 'Please configure your Gemini API key first'
            }), 400
        
        # Generate translation prompt
        language_name = SUPPORTED_LANGUAGES[target_language]
        prompt = f"""
        Translate the following historical text to {language_name} while preserving:
        1. All proper names, places, and historical terms (keep original or transliterate)
        2. Dates and numbers exactly as they appear
        3. The formal, educational tone
        4. Cultural context and meaning
        5. Markdown formatting if present
        
        Text to translate:
        {text}
        
        Provide only the translation without any additional commentary or explanations.
        """
        
        # Get translation from AI
        translated = generate_content(prompt, temperature=0.3, max_tokens=2048)
        
        if not translated:
            return jsonify({'error': 'Translation failed - empty response from AI'}), 500
        
        return jsonify({
            'original_text': text,
            'translated_text': translated,
            'target_language': language_name,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f" Translation error: {str(e)}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@translate_bp.route('/languages', methods=['GET'])
def get_languages():
    """
    Get list of supported languages
    
    Returns:
        {
            "languages": {
                "english": "English",
                ...
            },
            "count": 20
        }
    """
    return jsonify({
        'languages': SUPPORTED_LANGUAGES,
        'count': len(SUPPORTED_LANGUAGES)
    })

@translate_bp.route('/detect-language', methods=['POST'])
def detect_language():
    """
    Detect language of given text (using AI)
    
    Expected JSON:
        {
            "text": "Sample text to detect..."
        }
        
    Returns:
        {
            "text": "...",
            "detected_language": "...",
            "confidence": "high|medium|low"
        }
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not is_gemini_configured():
            return jsonify({
                'error': 'Language detection requires AI configuration'
            }), 400
        
        prompt = f"""
        Detect the language of the following text and respond with ONLY the language name in English (e.g., "English", "Spanish", "Hindi", etc.):
        
        {text[:500]}
        
        Language:
        """
        
        detected = generate_content(prompt, temperature=0.1, max_tokens=50)
        
        if detected:
            language_name = detected.strip()
            return jsonify({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'detected_language': language_name,
                'confidence': 'high' if len(text) > 50 else 'medium'
            })
        
        return jsonify({'error': 'Failed to detect language'}), 500
        
    except Exception as e:
        print(f" Language detection error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500
