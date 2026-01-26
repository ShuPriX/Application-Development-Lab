import os
import sys
import time

# --- DIAGNOSTIC CHECK ---
try:
    import langchain_core
    import langchain_community
    import langchain_ollama
    # We attempt to import the raw ollama library to list models
    import ollama
    print("✅ Core LangChain and Ollama libraries imported successfully")
except ImportError as e:
    print(f"\n❌ ERROR: Missing libraries. Details: {e}")

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import pandas as pd

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
VECTOR_DB_FOLDER = 'vector_db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store the Chain
qa_chain = None

def get_available_models():
    """Helper to list models currently available in Ollama."""
    try:
        models_info = ollama.list()
        # 'models' key contains a list of dicts, we want the 'name' field
        return [m['name'] for m in models_info.get('models', [])]
    except Exception as e:
        print(f"⚠️ Could not list Ollama models: {e}")
        return []

def load_document(file_path):
    """Detect file type and load content."""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            return loader.load()
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
            return [type('Document', (object,), {'page_content': text, 'metadata': {'source': file_path}})()]
    except Exception as e:
        print(f"Error loading document: {e}")
        return []
    return []

def process_document(file_path, model_name):
    global qa_chain
    
    # 0. Check available models
    available = get_available_models()
    print(f"🔍 Ollama sees these models: {available}")
    
    if model_name not in available:
        print(f"⚠️ Warning: Requested '{model_name}' not found in list. Trying anyway (Ollama might auto-pull or match fuzzy).")

    # 1. Load and Split Document
    docs = load_document(file_path)
    if not docs:
        raise ValueError("Could not read document content.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 2. Create Embeddings with Fallback
    print(f"Attempting to generate embeddings with: {model_name}")
    
    embeddings = None
    
    # Try the requested model first
    try:
        test_embeddings = OllamaEmbeddings(model=model_name)
        # Test a single query to ensure it works before processing the whole doc
        test_embeddings.embed_query("test connection")
        embeddings = test_embeddings
        print(f"✅ Embedding success with {model_name}")
    except Exception as e:
        print(f"❌ Failed to use {model_name} for embeddings: {e}")
        
        # Fallback Strategy: Try standard stable models
        fallbacks = ["llama3.2:latest", "llama3.2", "llama3"]
        for fb in fallbacks:
            if fb in available or fb == "llama3.2:latest":
                print(f"🔄 Attempting fallback to: {fb}")
                try:
                    test_embeddings = OllamaEmbeddings(model=fb)
                    test_embeddings.embed_query("test connection")
                    embeddings = test_embeddings
                    print(f"✅ Fallback success with {fb}")
                    break
                except:
                    continue
    
    if not embeddings:
        raise ValueError(f"Could not initialize embeddings with {model_name} or any fallbacks. Please check 'ollama list'.")
    
    # Create Vector Store (FAISS)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # 3. Setup LLM
    # We use the requested model for generation (even if we used a fallback for embeddings)
    llm = OllamaLLM(model=model_name)
    
    # 4. Create Chain using LCEL
    template = """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Chain created successfully using LCEL!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    model = request.form.get('model', 'llama3.2:latest')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            process_document(file_path, model)
            return jsonify({"message": f"Processed successfully! Ready to chat."}), 200
        except Exception as e:
            print(f"Error details: {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global qa_chain
    data = request.json
    user_query = data.get('query')
    
    if not qa_chain:
        return jsonify({"response": "Please upload a document first."})
    
    try:
        result = qa_chain.invoke(user_query)
        return jsonify({"response": result})
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)