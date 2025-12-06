"""
Summarization routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime

summarize_bp = Blueprint('summarize', __name__)

@summarize_bp.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Summarize historical content
    
    Expected JSON:
        {
            "text": "Long historical content to summarize...",
            "length": "short|medium|long"  (optional, default: "medium")
        }
        
    Returns:
        {
            "original_length": 1234,
            "summary": "...",
            "summary_length": 456,
            "compression_ratio": 0.37,
            "timestamp": "..."
        }
    """
    from utils.ai_utils import is_gemini_configured, generate_content
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        length = data.get('length', 'medium').lower()
        
        if not text:
            return jsonify({'error': 'Text to summarize is required'}), 400
        
        if length not in ['short', 'medium', 'long']:
            length = 'medium'
        
        # Check if AI is configured
        if not is_gemini_configured():
            return jsonify({
                'error': 'AI summarization requires API configuration',
                'suggestion': 'Please configure your Gemini API key first'
            }), 400
        
        # Determine sentence count based on length
        sentence_limits = {
            'short': '2-3 sentences',
            'medium': '4-5 sentences',
            'long': '8-10 sentences'
        }
        
        # Generate summarization prompt
        prompt = f"""
        Create a concise summary of this historical content:
        
        Guidelines:
        - Keep all essential facts, dates, and names
        - Maintain historical accuracy
        - Length: {sentence_limits[length]}
        - Preserve the most important information
        - Use clear, educational language
        - Maintain markdown formatting if needed
        
        Content to summarize:
        {text}
        
        Provide only the summary without additional commentary or preamble.
        """
        
        # Get summary from AI
        summary = generate_content(prompt, temperature=0.5, max_tokens=1024)
        
        if not summary:
            return jsonify({'error': 'Summarization failed - empty response from AI'}), 500
        
        # Calculate statistics
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return jsonify({
            'original_length': original_words,
            'summary': summary,
            'summary_length': summary_words,
            'compression_ratio': round(compression_ratio, 2),
            'length_type': length,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f" Summarization error: {str(e)}")
        return jsonify({'error': f'Summarization failed: {str(e)}'}), 500

@summarize_bp.route('/key-points', methods=['POST'])
def extract_key_points():
    """
    Extract key points from historical content
    
    Expected JSON:
        {
            "text": "Historical content...",
            "max_points": 5  (optional, default: 5)
        }
        
    Returns:
        {
            "key_points": ["Point 1", "Point 2", ...],
            "count": 5,
            "timestamp": "..."
        }
    """
    from utils.ai_utils import is_gemini_configured, generate_content
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        max_points = data.get('max_points', 5)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not isinstance(max_points, int) or max_points < 1 or max_points > 10:
            max_points = 5
        
        if not is_gemini_configured():
            return jsonify({
                'error': 'Key point extraction requires AI configuration'
            }), 400
        
        prompt = f"""
        Extract the {max_points} most important key points from this historical content.
        
        Guidelines:
        - Each point should be concise (1-2 sentences max)
        - Focus on facts, dates, and significance
        - Number each point (1., 2., 3., etc.)
        - Preserve historical accuracy
        
        Content:
        {text}
        
        Key Points:
        """
        
        response = generate_content(prompt, temperature=0.3, max_tokens=512)
        
        if not response:
            return jsonify({'error': 'Key point extraction failed'}), 500
        
        # Parse numbered points
        import re
        points = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)', response, re.DOTALL)
        points = [p.strip() for p in points if p.strip()]
        
        return jsonify({
            'key_points': points,
            'count': len(points),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f" Key point extraction error: {str(e)}")
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500
