"""
LlamaIndex Integration for Advanced Web Scraping
This module demonstrates how to use LlamaIndex to create searchable
indexes of scraped web content.

Prerequisites:
pip install llama-index llama-index-embeddings-huggingface --break-system-packages

LlamaIndex Features:
- Document indexing and retrieval
- Semantic search over scraped content
- Question answering from scraped data
- Multi-document synthesis
"""

from typing import List, Dict, Optional
import json

# Uncomment when llama-index is available:
# from llama_index.core import VectorStoreIndex, Document, Settings
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class LlamaIndexProcessor:
    """Use LlamaIndex to index and query scraped web content"""
    
    def __init__(self):
        """Initialize LlamaIndex processor"""
        self.available = self._check_availability()
        self.index = None
        self.documents = []
        
        if self.available:
            self._setup_embeddings()
    
    def _check_availability(self) -> bool:
        """Check if LlamaIndex is available"""
        try:
            # Uncomment when llama-index is available:
            # import llama_index
            # return True
            return False
        except ImportError:
            return False
    
    def _setup_embeddings(self):
        """Setup embedding model for semantic search"""
        # Uncomment when llama-index is available:
        # # Use a local embedding model (no API key needed)
        # embed_model = HuggingFaceEmbedding(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2"
        # )
        # Settings.embed_model = embed_model
        pass
    
    def add_scraped_content(self, scraped_data: Dict):
        """
        Add scraped content to the index
        
        Args:
            scraped_data: Dictionary containing scraped web data
        """
        if not self.available:
            print("LlamaIndex not available")
            return
        
        # Uncomment when llama-index is available:
        # # Create document from scraped data
        # text = scraped_data.get('text_content', '')
        # metadata = {
        #     'title': scraped_data.get('title', 'Untitled'),
        #     'url': scraped_data.get('url', ''),
        #     'scraped_at': scraped_data.get('scraped_at', ''),
        #     'word_count': len(text.split())
        # }
        # 
        # doc = Document(text=text, metadata=metadata)
        # self.documents.append(doc)
        # 
        # # Rebuild index with new document
        # self.index = VectorStoreIndex.from_documents(self.documents)
        pass
    
    def add_multiple_pages(self, scraped_pages: List[Dict]):
        """
        Add multiple scraped pages to the index
        
        Args:
            scraped_pages: List of scraped data dictionaries
        """
        if not self.available:
            print("LlamaIndex not available")
            return
        
        # Uncomment when llama-index is available:
        # documents = []
        # for page in scraped_pages:
        #     text = page.get('text_content', '')
        #     metadata = {
        #         'title': page.get('title', 'Untitled'),
        #         'url': page.get('url', ''),
        #         'word_count': len(text.split())
        #     }
        #     doc = Document(text=text, metadata=metadata)
        #     documents.append(doc)
        # 
        # self.documents.extend(documents)
        # self.index = VectorStoreIndex.from_documents(self.documents)
        pass
    
    def query(self, question: str) -> Dict:
        """
        Query the indexed content
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary with answer and source information
        """
        if not self.available or not self.index:
            return {
                'answer': 'LlamaIndex not available or no documents indexed',
                'sources': []
            }
        
        # Uncomment when llama-index is available:
        # # Create query engine
        # query_engine = self.index.as_query_engine(
        #     similarity_top_k=3,
        #     response_mode="compact"
        # )
        # 
        # # Execute query
        # response = query_engine.query(question)
        # 
        # # Extract sources
        # sources = []
        # for node in response.source_nodes:
        #     sources.append({
        #         'text': node.node.text[:200] + '...',
        #         'metadata': node.node.metadata,
        #         'score': node.score
        #     })
        # 
        # return {
        #     'answer': str(response),
        #     'sources': sources
        # }
        
        return {'answer': 'Not available', 'sources': []}
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search over indexed content
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if not self.available or not self.index:
            return []
        
        # Uncomment when llama-index is available:
        # retriever = self.index.as_retriever(similarity_top_k=top_k)
        # nodes = retriever.retrieve(query)
        # 
        # results = []
        # for node in nodes:
        #     results.append({
        #         'text': node.node.text,
        #         'metadata': node.node.metadata,
        #         'relevance_score': node.score
        #     })
        # 
        # return results
        
        return []
    
    def summarize_all(self) -> str:
        """
        Generate summary of all indexed content
        
        Returns:
            Summary text
        """
        if not self.available or not self.index:
            return "No content indexed"
        
        # Uncomment when llama-index is available:
        # query_engine = self.index.as_query_engine(response_mode="tree_summarize")
        # response = query_engine.query("Provide a comprehensive summary of all the content")
        # return str(response)
        
        return "Not available"
    
    def compare_sources(self, question: str) -> Dict:
        """
        Compare how different sources answer a question
        
        Args:
            question: Question to analyze across sources
            
        Returns:
            Comparison analysis
        """
        if not self.available or not self.index:
            return {'comparison': 'Not available'}
        
        # Uncomment when llama-index is available:
        # query_engine = self.index.as_query_engine(
        #     similarity_top_k=5,
        #     response_mode="compact"
        # )
        # 
        # response = query_engine.query(
        #     f"How do different sources address this question: {question}? "
        #     f"Compare and contrast the perspectives."
        # )
        # 
        # return {
        #     'question': question,
        #     'comparison': str(response),
        #     'sources_analyzed': len(response.source_nodes)
        # }
        
        return {'comparison': 'Not available'}
    
    def get_statistics(self) -> Dict:
        """Get statistics about indexed content"""
        return {
            'total_documents': len(self.documents),
            'indexed': self.index is not None,
            'available': self.available
        }


# Example usage scenarios

def example_research_assistant():
    """Example: Build a research assistant from scraped pages"""
    
    print("Research Assistant Example")
    print("-" * 50)
    
    # Mock scraped data
    pages = [
        {
            'title': 'Introduction to AI',
            'url': 'https://example.com/ai-intro',
            'text_content': 'Artificial intelligence is the simulation of human intelligence...'
        },
        {
            'title': 'Machine Learning Basics',
            'url': 'https://example.com/ml-basics',
            'text_content': 'Machine learning is a method of data analysis that automates...'
        },
        {
            'title': 'Deep Learning Guide',
            'url': 'https://example.com/dl-guide',
            'text_content': 'Deep learning is a subset of machine learning that uses neural networks...'
        }
    ]
    
    processor = LlamaIndexProcessor()
    
    if processor.available:
        # Index all pages
        print("Indexing pages...")
        processor.add_multiple_pages(pages)
        
        # Query the indexed content
        questions = [
            "What is the difference between AI and machine learning?",
            "How does deep learning relate to neural networks?",
            "What are the main applications of these technologies?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            result = processor.query(question)
            print(f"A: {result['answer']}")
            print(f"Sources: {len(result['sources'])} documents")
    else:
        print("LlamaIndex not available. Install with:")
        print("pip install llama-index llama-index-embeddings-huggingface --break-system-packages")


def example_semantic_search():
    """Example: Semantic search over scraped content"""
    
    print("\nSemantic Search Example")
    print("-" * 50)
    
    processor = LlamaIndexProcessor()
    
    if processor.available:
        # Mock indexed content
        processor.add_multiple_pages([
            {'text_content': 'Python is a high-level programming language...', 'url': 'url1'},
            {'text_content': 'JavaScript is essential for web development...', 'url': 'url2'},
            {'text_content': 'Machine learning models require data preprocessing...', 'url': 'url3'}
        ])
        
        # Search
        query = "programming languages for beginners"
        results = processor.semantic_search(query)
        
        print(f"Search: {query}")
        print(f"Found {len(results)} relevant results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata']['url']}")
            print(f"   Relevance: {result['relevance_score']:.2f}")
            print(f"   {result['text'][:100]}...")
    else:
        print("LlamaIndex not available")


# Flask integration

def create_llamaindex_routes(app):
    """
    Add LlamaIndex routes to Flask app
    
    Add to app.py:
    
    from llamaindex_integration import LlamaIndexProcessor, create_llamaindex_routes
    
    llamaindex_processor = LlamaIndexProcessor()
    create_llamaindex_routes(app)
    """
    
    from flask import request, jsonify
    
    processor = LlamaIndexProcessor()
    
    @app.route('/api/index/add', methods=['POST'])
    def add_to_index():
        """Add scraped content to index"""
        data = request.json
        scraped_data = data.get('scraped_data')
        
        if not scraped_data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        processor.add_scraped_content(scraped_data)
        
        return jsonify({
            'success': True,
            'statistics': processor.get_statistics()
        })
    
    @app.route('/api/index/query', methods=['POST'])
    def query_index():
        """Query indexed content"""
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'})
        
        result = processor.query(question)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    @app.route('/api/index/search', methods=['POST'])
    def semantic_search():
        """Semantic search"""
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        results = processor.semantic_search(query, top_k)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    @app.route('/api/index/stats', methods=['GET'])
    def get_stats():
        """Get index statistics"""
        return jsonify({
            'success': True,
            'statistics': processor.get_statistics()
        })


if __name__ == '__main__':
    print("LlamaIndex Integration Examples")
    print("=" * 50)
    print("\nTo use LlamaIndex:")
    print("1. Install: pip install llama-index llama-index-embeddings-huggingface --break-system-packages")
    print("2. Uncomment the import statements and code")
    print("3. Run the examples")
    
    print("\nFeatures:")
    print("- Semantic search over scraped content")
    print("- Natural language Q&A from indexed pages")
    print("- Multi-document synthesis")
    print("- Source comparison")
    print("- Local embedding models (no API keys needed)")
    
    # Run examples
    example_research_assistant()
    example_semantic_search()
