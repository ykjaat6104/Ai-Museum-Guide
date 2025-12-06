"""
Q&A routes for historical questions
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
import asyncio

qa_bp = Blueprint('qa', __name__)

# Global state (will be set by main app)
vector_index = None
text_map = None
smithsonian_api_key = None

def set_vector_db(index, t_map):
    """Set vector database for this blueprint"""
    global vector_index, text_map
    vector_index = index
    text_map = t_map

def set_museum_api_key(key):
    """Set museum API key"""
    global smithsonian_api_key
    smithsonian_api_key = key

async def get_ai_response(question):
    """
    Get comprehensive AI response for historical question
    
    Args:
        question: User's question string
        
    Returns:
        dict: Response with answer, source, and metadata
    """
    # Import only when function is called
    from utils.history_utils import is_historical_question, generate_history_prompt
    from utils.vector_utils import search_vector_db
    from utils.wikipedia_utils import search_and_summarize
    from utils.museum_utils import search_multiple_museums
    from utils.ai_utils import is_gemini_configured, generate_content
    
    # Check if question is historical
    if not is_historical_question(question):
        return {
            'response': "I specialize in world history and historical topics. Please ask about historical events, figures, civilizations, wars, cultural movements, or historical places from any time period and region.",
            'source': 'filter',
            'wikipedia_info': None,
            'museum_data': None
        }
    
    # Get context from vector database
    relevant_context = None
    if vector_index and text_map:
        contexts = search_vector_db(question, vector_index, text_map, k=3)
        if contexts:
            relevant_context = "\n\n".join(contexts)
    
    # Get Wikipedia information
    wikipedia_info = search_and_summarize(question)
    
    # Get museum artifacts (optional, don't block on failure)
    museum_data = None
    try:
        museum_results = search_multiple_museums(
            question,
            api_key=smithsonian_api_key,
            limit_per_source=3
        )
        if museum_results['total_count'] > 0:
            museum_data = museum_results
    except Exception as e:
        print(f"Museum search failed: {str(e)}")
    
    # Try AI response if configured
    if is_gemini_configured():
        try:
            prompt = generate_history_prompt(
                question,
                relevant_context,
                wikipedia_info,
                museum_data
            )
            
            ai_response = generate_content(prompt, temperature=0.7, max_tokens=2048)
            
            if ai_response:
                return {
                    'response': ai_response,
                    'source': 'ai',
                    'wikipedia_info': wikipedia_info,
                    'museum_data': museum_data,
                    'context_used': relevant_context is not None
                }
        except Exception as e:
            print(f" AI response error: {str(e)}")
    
    # Fallback to Wikipedia-based response
    fallback = generate_fallback_response(question, relevant_context, wikipedia_info)
    return {
        'response': fallback,
        'source': 'fallback',
        'wikipedia_info': wikipedia_info,
        'museum_data': museum_data,
        'context_used': relevant_context is not None
    }

@qa_bp.route('/ask', methods=['POST'])
def ask_question():
    """
    Main question answering endpoint
    
    Expected JSON:
        {
            "question": "What was the significance of the Roman Empire?"
        }
        
    Returns:
        {
            "question": "...",
            "answer": "...",
            "source": "ai|fallback|filter",
            "wikipedia_info": {...},
            "museum_data": {...},
            "timestamp": "..."
        }
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get AI response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(get_ai_response(question))
        finally:
            loop.close()
        
        response = {
            'question': question,
            'answer': result['response'],
            'source': result['source'],
            'wikipedia_info': result.get('wikipedia_info'),
            'museum_data': result.get('museum_data'),
            'context_used': result.get('context_used', False),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Question processing error: {str(e)}")
        return jsonify({'error': f'Failed to process question: {str(e)}'}), 500

@qa_bp.route('/quick-facts/<topic>', methods=['GET'])
def quick_facts(topic):
    """
    Get quick facts about a historical topic
    
    Args:
        topic: Historical topic name (URL parameter)
        
    Returns:
        Quick summary and key facts
    """
    try:
        # Get Wikipedia summary
        wikipedia_info = search_and_summarize(topic)
        
        if not wikipedia_info:
            return jsonify({'error': f'No information found for topic: {topic}'}), 404
        
        return jsonify({
            'topic': topic,
            'summary': wikipedia_info.get('extract', ''),
            'description': wikipedia_info.get('description', ''),
            'thumbnail': wikipedia_info.get('thumbnail', ''),
            'url': wikipedia_info.get('url', ''),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve facts: {str(e)}'}), 500

@qa_bp.route('/related/<topic>', methods=['GET'])
def related_topics(topic):
    """
    Get related historical topics
    
    Args:
        topic: Historical topic name (URL parameter)
        
    Returns:
        List of related topics with summaries
    """
    try:
        from utils.wikipedia_utils import get_related_articles
        
        related = get_related_articles(topic, limit=5)
        
        return jsonify({
            'topic': topic,
            'related_topics': related,
            'count': len(related),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve related topics: {str(e)}'}), 500
