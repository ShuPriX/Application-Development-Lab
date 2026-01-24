"""
Configuration file for the Web Scraper application
Modify these settings to customize the scraper behavior
"""

# Flask Application Settings
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'SECRET_KEY': 'your-secret-key-here'  # Change this in production
}

# Scraping Settings
SCRAPER_CONFIG = {
    # Request timeout in seconds
    'TIMEOUT': 10,
    
    # User agent string
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    
    # Maximum number of retries for failed requests
    'MAX_RETRIES': 3,
    
    # Delay between retries (seconds)
    'RETRY_DELAY': 2,
    
    # Maximum content length to process (in bytes)
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # 10MB
}

# Content Extraction Limits
EXTRACTION_LIMITS = {
    # Maximum number of links to extract
    'MAX_LINKS': 50,
    
    # Maximum number of images to extract
    'MAX_IMAGES': 20,
    
    # Maximum number of paragraphs to process
    'MAX_PARAGRAPHS': 100,
    
    # Maximum text content length for processing
    'MAX_TEXT_LENGTH': 50000,  # characters
}

# Batch Scraping Settings
BATCH_CONFIG = {
    # Maximum number of URLs to scrape in one batch
    'MAX_BATCH_SIZE': 5,
    
    # Delay between scraping each URL (seconds) - be respectful!
    'BATCH_DELAY': 1,
    
    # Maximum concurrent requests (not implemented yet, placeholder)
    'MAX_CONCURRENT': 3,
}

# Search Settings
SEARCH_CONFIG = {
    # Maximum number of search results to return
    'MAX_SEARCH_RESULTS': 5,
    
    # Search engine to use
    'SEARCH_ENGINE': 'duckduckgo',  # Options: 'duckduckgo', 'google' (requires API)
    
    # Whether to automatically scrape first search result
    'AUTO_SCRAPE_FIRST': True,
}

# LLM Processing Settings
LLM_CONFIG = {
    # Maximum text length to send to LLM (for summarization)
    'MAX_LLM_INPUT': 3000,  # characters
    
    # Number of key points to extract
    'KEY_POINTS_COUNT': 5,
    
    # Number of main topics to identify
    'MAIN_TOPICS_COUNT': 10,
    
    # Summary text maximum length
    'SUMMARY_MAX_LENGTH': 300,  # characters
}

# Ollama Settings (if using Ollama)
OLLAMA_CONFIG = {
    'ENABLED': False,  # Set to True to enable Ollama
    'MODEL': 'llama2',  # Model to use
    'API_URL': 'http://localhost:11434',  # Ollama API endpoint
    'TIMEOUT': 30,  # Timeout for Ollama requests
}

# LlamaIndex Settings (if using LlamaIndex)
LLAMAINDEX_CONFIG = {
    'ENABLED': False,  # Set to True to enable LlamaIndex
    'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
    'CHUNK_SIZE': 512,
    'CHUNK_OVERLAP': 50,
    'SIMILARITY_TOP_K': 3,
}

# Logging Settings
LOGGING_CONFIG = {
    'LEVEL': 'INFO',  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_FILE': 'scraper.log',  # Set to None to disable file logging
}

# Cache Settings (for future implementation)
CACHE_CONFIG = {
    'ENABLED': False,  # Set to True to enable caching
    'CACHE_DIR': './cache',
    'CACHE_EXPIRY': 3600,  # Cache expiry in seconds (1 hour)
}

# Rate Limiting (for future implementation)
RATE_LIMIT_CONFIG = {
    'ENABLED': False,
    'REQUESTS_PER_MINUTE': 60,
    'REQUESTS_PER_HOUR': 1000,
}

# Content Filtering
CONTENT_FILTER = {
    # HTML elements to remove during scraping
    'REMOVE_TAGS': ['script', 'style', 'nav', 'footer', 'header', 'aside'],
    
    # Minimum paragraph length to include
    'MIN_PARAGRAPH_LENGTH': 50,  # characters
    
    # Whether to extract only main content (experimental)
    'EXTRACT_MAIN_CONTENT': True,
}

# Export Settings
EXPORT_CONFIG = {
    'FORMATS': ['json', 'csv', 'txt'],  # Supported export formats
    'EXPORT_DIR': './exports',
    'INCLUDE_METADATA': True,
}

# Security Settings
SECURITY_CONFIG = {
    # Maximum URL length to accept
    'MAX_URL_LENGTH': 2048,
    
    # Allowed URL schemes
    'ALLOWED_SCHEMES': ['http', 'https'],
    
    # Blocked domains (for safety)
    'BLOCKED_DOMAINS': [],
    
    # Whether to validate SSL certificates
    'VERIFY_SSL': True,
}

# Frontend Settings
FRONTEND_CONFIG = {
    # Results per page
    'RESULTS_PER_PAGE': 10,
    
    # Theme
    'THEME': 'default',  # Options: 'default', 'dark', 'light'
    
    # Enable animations
    'ANIMATIONS': True,
}

# Database Settings (for future implementation)
DATABASE_CONFIG = {
    'ENABLED': False,
    'TYPE': 'sqlite',  # Options: 'sqlite', 'postgresql', 'mysql'
    'DATABASE': 'scraper.db',
    'HOST': 'localhost',
    'PORT': 5432,
    'USER': 'scraper',
    'PASSWORD': '',
}


# Helper function to get configuration
def get_config(category: str, key: str = None):
    """
    Get configuration value
    
    Args:
        category: Configuration category (e.g., 'FLASK_CONFIG')
        key: Specific key within category (optional)
    
    Returns:
        Configuration value or entire category
    
    Example:
        >>> get_config('FLASK_CONFIG', 'PORT')
        5000
        >>> get_config('SCRAPER_CONFIG')
        {'TIMEOUT': 10, ...}
    """
    config_map = {
        'FLASK': FLASK_CONFIG,
        'SCRAPER': SCRAPER_CONFIG,
        'EXTRACTION': EXTRACTION_LIMITS,
        'BATCH': BATCH_CONFIG,
        'SEARCH': SEARCH_CONFIG,
        'LLM': LLM_CONFIG,
        'OLLAMA': OLLAMA_CONFIG,
        'LLAMAINDEX': LLAMAINDEX_CONFIG,
        'LOGGING': LOGGING_CONFIG,
        'CACHE': CACHE_CONFIG,
        'RATE_LIMIT': RATE_LIMIT_CONFIG,
        'FILTER': CONTENT_FILTER,
        'EXPORT': EXPORT_CONFIG,
        'SECURITY': SECURITY_CONFIG,
        'FRONTEND': FRONTEND_CONFIG,
        'DATABASE': DATABASE_CONFIG,
    }
    
    category_config = config_map.get(category)
    
    if category_config is None:
        return None
    
    if key is None:
        return category_config
    
    return category_config.get(key)


# Example usage in app.py:
# from config import SCRAPER_CONFIG, FLASK_CONFIG
# 
# timeout = SCRAPER_CONFIG['TIMEOUT']
# app.config['SECRET_KEY'] = FLASK_CONFIG['SECRET_KEY']
