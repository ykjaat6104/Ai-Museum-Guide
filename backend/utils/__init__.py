"""
Utility modules for AI Museum Guide backend

Note: This module uses lazy imports to reduce memory footprint.
Import specific functions from submodules as needed.
"""

# Empty __init__ to prevent automatic loading of heavy dependencies
# Import from submodules directly: from utils.ai_utils import setup_gemini

__all__ = [
    'ai_utils',
    'vector_utils',
    'wikipedia_utils',
    'museum_utils',
    'history_utils'
]
