"""
Configuration management for AI Museum Guide Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root directory (optional - for local dev)
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Server
    PORT = int(os.getenv('FLASK_PORT', 5000))
    HOST = '0.0.0.0'
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    SMITHSONIAN_API_KEY = os.getenv('SMITHSONIAN_API_KEY', '')
    
    # Database and Cache
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', './data/faiss_index')
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', 50))
    RATE_LIMIT_PER_DAY = int(os.getenv('RATE_LIMIT_PER_DAY', 1500))
    
    # Content Ingestion
    AUTO_POPULATE_FAISS = os.getenv('AUTO_POPULATE_FAISS', 'True').lower() == 'true'
    WIKIPEDIA_ARTICLES_LIMIT = int(os.getenv('WIKIPEDIA_ARTICLES_LIMIT', 5000))
    
    # File Paths
    DATA_DIR = './data'
    FAISS_INDEX_FILE = os.path.join(DATA_DIR, 'faiss_index.bin')
    TEXT_MAP_FILE = os.path.join(DATA_DIR, 'faiss_text_map.json')
    GENERATED_IMAGES_DIR = os.path.join(DATA_DIR, 'generated_images')
    
    # AI Models
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    GEMINI_MODELS = [
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-pro',
        'gemini-1.0-pro',
        'models/gemini-pro',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro'
    ]
    
    # Performance
    CACHE_TIMEOUT = 3600  # 1 hour
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
