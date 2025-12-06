"""
AI Museum Guide - Consolidated Backend API
Version 2.0.0

A comprehensive Flask API for worldwide historical and museum exploration
with Gemini AI integration, Wikipedia data, and museum collection access.
"""
from flask import Flask, jsonify
from flask_cors import CORS
import os
import nest_asyncio
from config import get_config

# Configure environment
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Reduce memory usage
nest_asyncio.apply()

def create_app():
    """Application factory pattern"""
    # Import config here to avoid loading heavy modules
    from config import get_config
    config = get_config()
    
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": config.CORS_ORIGINS,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Create data directory
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.GENERATED_IMAGES_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("PASTPORTALS - Backend Server")
    print("="*60)
    print(f"\nEnvironment: {config.FLASK_ENV}")
    
    # Import routes only when needed
    from routes import (
        qa_bp, translate_bp, summarize_bp, museum_bp, config_bp,
        qa_set_vector_db, qa_set_museum_key,
        museum_set_api_key, config_set_vector_db
    )
    
    # Skip heavy models in production to save memory
    if config.FLASK_ENV == 'production':
        print("   Mode: Production (lightweight)")
        print("   AI Models: Using Gemini API only")
        print("   Vector DB: Disabled (uses Wikipedia API)")
        qa_set_vector_db(None, None)
        config_set_vector_db(None, None)
    else:
        # Only load heavy modules in development
        from utils.vector_utils import load_vector_db
        from utils.ai_utils import get_embeddings_model
        
        embeddings_model = get_embeddings_model(config.EMBEDDING_MODEL)
        embeddings_status = "Loaded" if embeddings_model else "Failed"
        print(f"   Embeddings Model: {embeddings_status}")
        
        vector_index, text_map = load_vector_db(
            config.FAISS_INDEX_FILE,
            config.TEXT_MAP_FILE
        )
        
        if vector_index and text_map:
            vector_status = f"Loaded ({vector_index.ntotal} vectors)"
        else:
            vector_status = "Empty (will use online sources)"
        print(f"   Vector Database: {vector_status}")
        
        qa_set_vector_db(vector_index, text_map)
        config_set_vector_db(vector_index, text_map)
    
    # Configure AI if API key is available
    from utils.ai_utils import setup_gemini
    gemini_key = config.GEMINI_API_KEY
    if gemini_key:
        ai_configured = setup_gemini(gemini_key)
        ai_status = "Configured" if ai_configured else "Configuration failed"
    else:
        ai_status = "Not configured (use /configure endpoint)"
    print(f"   Gemini AI: {ai_status}")
    
    # Set museum API keys
    if config.SMITHSONIAN_API_KEY:
        museum_set_api_key(config.SMITHSONIAN_API_KEY)
        qa_set_museum_key(config.SMITHSONIAN_API_KEY)
        museum_status = "Configured"
    else:
        museum_status = "No API key (public access only)"
    print(f"   Museum APIs: {museum_status}")
    
    print(f"   Wikipedia API: Ready")
    print("="*60 + "\n")
    
    # Register blueprints
    app.register_blueprint(config_bp, url_prefix='/api')
    app.register_blueprint(qa_bp, url_prefix='/api')
    app.register_blueprint(translate_bp, url_prefix='/api')
    app.register_blueprint(summarize_bp, url_prefix='/api')
    app.register_blueprint(museum_bp, url_prefix='/api/museum')
    
    # Root endpoint
    @app.route('/')
    def index():
        return jsonify({
            'name': 'PastPortals API',
            'version': '2.0.0',
            'description': 'Worldwide historical and museum exploration powered by AI',
            'endpoints': {
                'health': '/api/health',
                'configure': '/api/configure',
                'ask': '/api/ask',
                'translate': '/api/translate',
                'summarize': '/api/summarize',
                'museum_search': '/api/museum/search',
                'collections': '/api/museum/collections'
            },
            'documentation': 'https://github.com/ykjaat6104/Ai-Museum-Guide',
            'status': 'operational'
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
            'message': 'The requested resource does not exist',
            'available_endpoints': [
                '/api/health', '/api/configure', '/api/ask',
                '/api/translate', '/api/summarize', '/api/museum/search'
            ]
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please try again.'
        }), 500
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad request',
            'message': 'Invalid request format or parameters'
        }), 400
    
    print("\n" + "="*60)
    print(f"Server Ready - http://{config.HOST}:{config.PORT}")
    print(f"Environment: {config.FLASK_ENV}")
    print(f"CORS Origins: {', '.join(config.CORS_ORIGINS)}")
    print("="*60 + "\n")
    
    return app

# Create app instance for gunicorn
app = create_app()

if __name__ == '__main__':
    config = get_config()
    
    # Run development server
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )