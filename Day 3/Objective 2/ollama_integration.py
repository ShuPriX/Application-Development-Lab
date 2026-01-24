"""
Ollama Integration for Advanced LLM Processing
This module demonstrates how to integrate Ollama for more sophisticated
content analysis and summarization.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull a model: ollama pull llama2
3. Install Python client: pip install ollama --break-system-packages

Note: Ollama runs locally, so you have full control over your data
and don't need API keys or internet connection after model download.
"""

import json
from typing import Dict, List, Optional

# Uncomment when ollama is available:
# import ollama


class OllamaProcessor:
    """Process scraped content using Ollama for advanced LLM capabilities"""
    
    def __init__(self, model: str = "llama2"):
        """
        Initialize Ollama processor
        
        Args:
            model: The Ollama model to use (e.g., 'llama2', 'mistral', 'codellama')
        """
        self.model = model
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            # Uncomment when ollama is available:
            # ollama.list()
            # return True
            return False
        except:
            return False
    
    def summarize(self, scraped_data: Dict) -> Dict:
        """
        Generate advanced summary using Ollama
        
        Args:
            scraped_data: Dictionary containing scraped web data
            
        Returns:
            Dictionary with enhanced summary and insights
        """
        if not self.available:
            return self._fallback_summary(scraped_data)
        
        text_content = scraped_data.get('text_content', '')
        title = scraped_data.get('title', 'Untitled')
        
        # Truncate content if too long (Ollama has context limits)
        max_chars = 3000
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + "..."
        
        prompt = f"""
        Analyze the following web page content and provide:
        1. A concise 2-3 sentence summary
        2. 5 key points or insights
        3. The main topics covered
        4. The target audience
        5. The content type (article, blog, documentation, etc.)
        
        Title: {title}
        
        Content:
        {text_content}
        
        Provide the analysis in JSON format with keys: summary, key_points, topics, audience, content_type
        """
        
        try:
            # Uncomment when ollama is available:
            # response = ollama.generate(model=self.model, prompt=prompt)
            # result_text = response['response']
            
            # # Try to parse JSON from response
            # try:
            #     result = json.loads(result_text)
            # except:
            #     # If not valid JSON, create structured response
            #     result = {
            #         'summary': result_text[:500],
            #         'key_points': [],
            #         'topics': [],
            #         'audience': 'General',
            #         'content_type': 'Unknown'
            #     }
            
            # return result
            
            # Placeholder return for when Ollama is not available
            return self._fallback_summary(scraped_data)
            
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._fallback_summary(scraped_data)
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with extracted entities
        """
        if not self.available:
            return {'entities': []}
        
        prompt = f"""
        Extract named entities from this text. Identify:
        - People
        - Organizations
        - Locations
        - Dates
        - Technologies
        
        Text: {text[:1000]}
        
        Return as JSON with keys: people, organizations, locations, dates, technologies
        """
        
        try:
            # Uncomment when ollama is available:
            # response = ollama.generate(model=self.model, prompt=prompt)
            # result = json.loads(response['response'])
            # return result
            
            return {'entities': []}
        except:
            return {'entities': []}
    
    def compare_content(self, contents: List[Dict]) -> Dict:
        """
        Compare multiple scraped contents and find common themes
        
        Args:
            contents: List of scraped data dictionaries
            
        Returns:
            Comparison analysis
        """
        if not self.available or not contents:
            return {'comparison': 'Not available'}
        
        # Create summaries of each content
        summaries = []
        for content in contents[:3]:  # Limit to 3 for context window
            title = content.get('title', 'Unknown')
            text = content.get('text_content', '')[:500]
            summaries.append(f"Title: {title}\nContent: {text}\n")
        
        combined = "\n---\n".join(summaries)
        
        prompt = f"""
        Compare these web page contents and identify:
        1. Common themes
        2. Unique perspectives
        3. Contradictions or differences
        4. Overall consensus (if any)
        
        Contents:
        {combined}
        
        Provide analysis in JSON format with keys: common_themes, unique_perspectives, differences, consensus
        """
        
        try:
            # Uncomment when ollama is available:
            # response = ollama.generate(model=self.model, prompt=prompt)
            # result = json.loads(response['response'])
            # return result
            
            return {'comparison': 'Not available'}
        except:
            return {'comparison': 'Not available'}
    
    def generate_questions(self, scraped_data: Dict) -> List[str]:
        """
        Generate relevant questions based on the content
        
        Args:
            scraped_data: Scraped web data
            
        Returns:
            List of generated questions
        """
        if not self.available:
            return []
        
        text = scraped_data.get('text_content', '')[:1500]
        
        prompt = f"""
        Based on this content, generate 5 thought-provoking questions that:
        - Test understanding of the material
        - Encourage critical thinking
        - Explore implications
        
        Content: {text}
        
        Return as JSON array of questions.
        """
        
        try:
            # Uncomment when ollama is available:
            # response = ollama.generate(model=self.model, prompt=prompt)
            # questions = json.loads(response['response'])
            # return questions if isinstance(questions, list) else []
            
            return []
        except:
            return []
    
    def _fallback_summary(self, scraped_data: Dict) -> Dict:
        """Fallback summary when Ollama is not available"""
        paragraphs = scraped_data.get('paragraphs', [])
        headings = scraped_data.get('headings', {})
        
        return {
            'summary': paragraphs[0][:300] if paragraphs else 'No content available',
            'key_points': paragraphs[:3],
            'topics': headings.get('h2', [])[:5],
            'audience': 'General',
            'content_type': 'Web Page',
            'note': 'Using fallback processor - Ollama not available'
        }


# Example usage functions

def example_single_page_analysis():
    """Example: Analyze a single scraped page with Ollama"""
    
    # Mock scraped data
    scraped_data = {
        'url': 'https://example.com',
        'title': 'Introduction to Machine Learning',
        'text_content': 'Machine learning is a subset of artificial intelligence...',
        'paragraphs': ['Machine learning enables computers to learn...'],
        'headings': {'h2': ['What is ML?', 'Applications', 'Future']}
    }
    
    processor = OllamaProcessor(model='llama2')
    
    if processor.available:
        summary = processor.summarize(scraped_data)
        print("Summary:", summary)
        
        entities = processor.extract_entities(scraped_data['text_content'])
        print("Entities:", entities)
        
        questions = processor.generate_questions(scraped_data)
        print("Questions:", questions)
    else:
        print("Ollama not available. Using fallback processing.")


def example_batch_comparison():
    """Example: Compare multiple scraped pages"""
    
    scraped_pages = [
        {'title': 'AI News 1', 'text_content': 'Recent advances in AI...'},
        {'title': 'AI News 2', 'text_content': 'AI ethics concerns...'},
        {'title': 'AI News 3', 'text_content': 'AI in healthcare...'}
    ]
    
    processor = OllamaProcessor()
    
    if processor.available:
        comparison = processor.compare_content(scraped_pages)
        print("Comparison:", comparison)
    else:
        print("Ollama not available.")


# Integration with main Flask app

def integrate_with_flask(app):
    """
    Example of how to integrate Ollama processor with Flask app
    
    Add this to app.py:
    
    from ollama_integration import OllamaProcessor
    
    ollama_processor = OllamaProcessor(model='llama2')
    
    @app.route('/api/ollama-analyze', methods=['POST'])
    def ollama_analyze():
        data = request.json
        scraped_data = data.get('scraped_data')
        
        if ollama_processor.available:
            result = {
                'summary': ollama_processor.summarize(scraped_data),
                'entities': ollama_processor.extract_entities(scraped_data['text_content']),
                'questions': ollama_processor.generate_questions(scraped_data)
            }
            return jsonify({'success': True, 'analysis': result})
        else:
            return jsonify({'success': False, 'error': 'Ollama not available'})
    """
    pass


if __name__ == '__main__':
    print("Ollama Integration Example")
    print("=" * 50)
    print("\nTo use this module:")
    print("1. Install Ollama from https://ollama.ai/")
    print("2. Pull a model: ollama pull llama2")
    print("3. Uncomment the ollama imports and code")
    print("4. Run your Flask app with Ollama integration")
    print("\nAvailable models:")
    print("- llama2: General purpose")
    print("- mistral: Fast and capable")
    print("- codellama: Code-focused")
    print("- llama2:70b: Larger, more capable (requires more RAM)")
    
    # Run example (will use fallback if Ollama not available)
    example_single_page_analysis()
