"""
Configuration and health check routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
import os
from utils import setup_gemini, is_gemini_configured, get_embeddings_model

config_bp = Blueprint('config', __name__)

# Global state (will be set by main app)
vector_index = None
text_map = None

def set_vector_db(index, t_map):
    """Set vector database for health checks"""
    global vector_index, text_map
    vector_index = index
    text_map = t_map

@config_bp.route('/health', methods=['GET'])
def health():
    """
    System health check endpoint
    
    Returns:
        {
            "status": "online",
            "timestamp": "...",
            "ai_configured": true,
            "services": {
                "gemini_ai": true,
                "wikipedia": true,
                "embeddings": true,
                "vector_db": true,
                "museum_api": true
            },
            "version": "2.0.0"
        }
    """
    vector_db_exists = vector_index is not None and text_map is not None
    vector_count = vector_index.ntotal if vector_db_exists else 0
    
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'ai_configured': is_gemini_configured(),
        'services': {
            'gemini_ai': is_gemini_configured(),
            'wikipedia': True,
            'embeddings': get_embeddings_model() is not None,
            'vector_db': vector_db_exists,
            'vector_count': vector_count,
            'museum_api': True
        },
        'version': '2.0.0',
        'environment': os.getenv('FLASK_ENV', 'development')
    })

@config_bp.route('/configure', methods=['POST'])
def configure_api():
    """
    Configure Gemini AI API key
    
    Expected JSON:
        {
            "api_key": "your-gemini-api-key"
        }
        
    Returns:
        {
            "message": "API configured successfully",
            "ai_enabled": true
        }
    """
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        # Setup Gemini with the provided key
        success = setup_gemini(api_key)
        
        if success:
            return jsonify({
                'message': 'API configured successfully',
                'ai_enabled': True,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Failed to configure API. Please check your API key.',
                'ai_enabled': False,
                'suggestion': 'Verify your Gemini API key at https://makersuite.google.com/app/apikey'
            }), 400
            
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        return jsonify({'error': f'Configuration error: {str(e)}'}), 500

@config_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get detailed system status
    
    Returns:
        Comprehensive system status information
    """
    try:
        from utils import get_vector_db_stats
        
        vector_stats = get_vector_db_stats(vector_index, text_map) if vector_index else {
            'total_vectors': 0,
            'dimension': 0,
            'text_entries': 0,
            'status': 'not_initialized'
        }
        
        embeddings = get_embeddings_model()
        
        return jsonify({
            'system': {
                'status': 'operational',
                'uptime': 'N/A',  # Could track this with startup time
                'version': '2.0.0',
                'environment': os.getenv('FLASK_ENV', 'development')
            },
            'ai': {
                'gemini_configured': is_gemini_configured(),
                'embeddings_loaded': embeddings is not None,
                'model': 'all-mpnet-base-v2' if embeddings else None
            },
            'database': {
                'vector_db': vector_stats,
                'total_documents': vector_stats['text_entries']
            },
            'apis': {
                'wikipedia': {'status': 'available', 'rate_limit': None},
                'smithsonian': {'status': 'available', 'requires_key': False},
                'europeana': {'status': 'requires_registration'}
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Status check error: {str(e)}")
        return jsonify({'error': f'Failed to retrieve status: {str(e)}'}), 500

@config_bp.route('/capabilities', methods=['GET'])
def get_capabilities():
    """
    Get list of available features and capabilities
    
    Returns:
        List of features supported by the API
    """
    capabilities = {
        'core_features': {
            'historical_qa': {
                'enabled': True,
                'description': 'AI-powered Q&A for worldwide history',
                'requires_ai': True
            },
            'translation': {
                'enabled': is_gemini_configured(),
                'description': 'Translate content to 18+ languages',
                'requires_ai': True,
                'languages_supported': 18
            },
            'summarization': {
                'enabled': is_gemini_configured(),
                'description': 'Summarize historical content',
                'requires_ai': True
            },
            'museum_search': {
                'enabled': True,
                'description': 'Search museum collections worldwide',
                'requires_ai': False,
                'museums': ['smithsonian']
            },
            'wikipedia_integration': {
                'enabled': True,
                'description': 'Real-time Wikipedia data',
                'requires_ai': False
            },
            'vector_search': {
                'enabled': vector_index is not None,
                'description': 'Semantic search in knowledge base',
                'requires_ai': False
            }
        },
        'advanced_features': {
            'multi_source_search': True,
            'context_aware_responses': True,
            'fallback_mode': True,
            'batch_translation': False,
            'image_recognition': False,
            'voice_interface': False
        }
    }
    
    return jsonify({
        'capabilities': capabilities,
        'timestamp': datetime.now().isoformat()
    })
