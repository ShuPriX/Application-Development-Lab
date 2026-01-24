from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import nest_asyncio
from pathlib import Path
import json

nest_asyncio.apply()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt', 'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for index and query engine
current_index = None
current_query_engine = None
current_documents = []

# Available LLM models
AVAILABLE_MODELS = {
    'ollama': [
        {'id': 'gemma3:12b', 'name': 'Gemma 3 (12B)', 'provider': 'ollama'},
        {'id': 'llama3.2', 'name': 'Llama 3.2 (3B)', 'provider': 'ollama'},
        {'id': 'llama3.2:1b', 'name': 'Llama 3.2 (1B)', 'provider': 'ollama'},
        {'id': 'gemma2:2b', 'name': 'Gemma 2 (2B)', 'provider': 'ollama'},
        {'id': 'gemma2:9b', 'name': 'Gemma 2 (9B)', 'provider': 'ollama'},
        {'id': 'phi3:mini', 'name': 'Phi 3 Mini', 'provider': 'ollama'},
        {'id': 'mistral:7b', 'name': 'Mistral 7B', 'provider': 'ollama'},
    ]
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_embeddings():
    """Initialize the embedding model"""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    return embed_model

def initialize_llm(model_id, provider, api_key=None):
    """Initialize the LLM based on provider and model"""
    if provider == 'ollama':
        llm = Ollama(model=model_id, request_timeout=120.0)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    Settings.llm = llm
    return llm

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'RAG API is running'})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available LLM models"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'success': True
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process document files"""
    global current_index, current_documents
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'success': False}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'success': False}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
            'success': False
        }), 400
    
    try:
        # Clear previous uploads
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Save new file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and index documents
        embed_model = initialize_embeddings()
        documents = SimpleDirectoryReader(input_dir=UPLOAD_FOLDER).load_data()
        current_documents = documents
        
        # Create vector index
        current_index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model
        )
        
        return jsonify({
            'success': True,
            'message': f'File "{filename}" uploaded and indexed successfully',
            'filename': filename,
            'num_documents': len(documents)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/query', methods=['POST'])
def query_documents():
    """Query the indexed documents"""
    global current_index, current_query_engine
    
    if current_index is None:
        return jsonify({
            'error': 'No documents uploaded. Please upload a document first.',
            'success': False
        }), 400
    
    data = request.json
    query_text = data.get('query', '').strip()
    model_id = data.get('model_id', 'llama3.2')
    provider = data.get('provider', 'ollama')
    api_key = data.get('api_key', '')
    
    if not query_text:
        return jsonify({
            'error': 'Query text is required',
            'success': False
        }), 400
    
    try:
        # Initialize LLM
        llm = initialize_llm(model_id, provider, api_key if api_key else None)
        
        # Create or update query engine
        current_query_engine = current_index.as_query_engine(llm=llm)
        
        # Execute query
        response = current_query_engine.query(query_text)
        
        # Extract source information
        source_nodes = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes[:3]:  # Top 3 sources
                source_nodes.append({
                    'text': node.node.text[:200] + '...' if len(node.node.text) > 200 else node.node.text,
                    'score': float(node.score) if hasattr(node, 'score') else None
                })
        
        return jsonify({
            'success': True,
            'response': str(response),
            'sources': source_nodes,
            'model_used': f"{provider}/{model_id}"
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_documents():
    """Clear uploaded documents and reset index"""
    global current_index, current_query_engine, current_documents
    
    try:
        # Clear uploaded files
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Reset global variables
        current_index = None
        current_query_engine = None
        current_documents = []
        
        return jsonify({
            'success': True,
            'message': 'All documents cleared successfully'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/')
def serve_frontend():
    """Serve the frontend"""
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    print("=" * 50)
    print("RAG Document Query System")
    print("=" * 50)
    print("Starting Flask server...")
    print("Server will be available at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
