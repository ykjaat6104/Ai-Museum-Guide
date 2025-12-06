"""
Museum API routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from utils import search_smithsonian, search_multiple_museums, format_museum_response

museum_bp = Blueprint('museum', __name__)

# Global state
smithsonian_api_key = None

def set_api_key(key):
    """Set Smithsonian API key"""
    global smithsonian_api_key
    smithsonian_api_key = key

@museum_bp.route('/search', methods=['POST'])
def search_museums():
    """
    Search museum collections for artifacts
    
    Expected JSON:
        {
            "query": "ancient egyptian pottery",
            "limit": 10,
            "sources": ["smithsonian"]  (optional)
        }
        
    Returns:
        {
            "query": "...",
            "results": {...},
            "total_count": 15,
            "sources_used": ["smithsonian"],
            "timestamp": "..."
        }
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = data.get('limit', 10)
        requested_sources = data.get('sources', ['smithsonian'])
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            limit = 10
        
        # Search museums
        results = search_multiple_museums(
            query,
            api_key=smithsonian_api_key,
            limit_per_source=limit
        )
        
        return jsonify({
            'query': query,
            'results': results,
            'total_count': results['total_count'],
            'sources_used': ['smithsonian'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Museum search error: {str(e)}")
        return jsonify({'error': f'Museum search failed: {str(e)}'}), 500

@museum_bp.route('/artifact/<artifact_id>', methods=['GET'])
def get_artifact(artifact_id):
    """
    Get detailed information about a specific artifact
    
    Args:
        artifact_id: Artifact ID (URL parameter)
        
    Returns:
        Detailed artifact information
    """
    try:
        from utils.museum_utils import get_smithsonian_object
        
        artifact = get_smithsonian_object(artifact_id, api_key=smithsonian_api_key)
        
        if not artifact:
            return jsonify({'error': f'Artifact not found: {artifact_id}'}), 404
        
        return jsonify({
            'artifact': artifact,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f" Artifact retrieval error: {str(e)}")
        return jsonify({'error': f'Failed to retrieve artifact: {str(e)}'}), 500

@museum_bp.route('/collections', methods=['GET'])
def get_collections():
    """
    Get list of available museum collections
    
    Returns:
        List of museums and collections available through API
    """
    collections = {
        'smithsonian': {
            'name': 'Smithsonian Institution',
            'description': 'World\'s largest museum, education, and research complex',
            'collections': [
                'National Museum of Natural History',
                'National Air and Space Museum',
                'National Museum of American History',
                'Smithsonian American Art Museum',
                'National Portrait Gallery',
                'And 14 more museums'
            ],
            'total_items': '155 million+',
            'open_access': True,
            'api_docs': 'https://api.si.edu/openaccess'
        },
        'europeana': {
            'name': 'Europeana',
            'description': 'European cultural heritage collections',
            'collections': ['Art', 'Archaeology', 'Fashion', 'Music', 'Photography'],
            'total_items': '50 million+',
            'open_access': True,
            'api_docs': 'https://pro.europeana.eu/page/apis',
            'status': 'Requires API key registration'
        }
    }
    
    return jsonify({
        'collections': collections,
        'available_count': len(collections),
        'timestamp': datetime.now().isoformat()
    })

@museum_bp.route('/categories', methods=['GET'])
def get_categories():
    """
    Get artifact categories for filtering
    
    Returns:
        List of artifact categories
    """
    categories = {
        'art': 'Paintings, Sculptures, Art Objects',
        'archaeology': 'Ancient Artifacts, Excavation Finds',
        'history': 'Historical Objects, Documents',
        'cultural': 'Cultural Heritage, Traditional Items',
        'natural_history': 'Fossils, Minerals, Specimens',
        'technology': 'Inventions, Tools, Machines',
        'decorative_arts': 'Furniture, Textiles, Ceramics',
        'photography': 'Historical Photographs',
        'manuscripts': 'Historical Documents, Letters',
        'numismatics': 'Coins, Currency, Medals'
    }
    
    return jsonify({
        'categories': categories,
        'count': len(categories)
    })
