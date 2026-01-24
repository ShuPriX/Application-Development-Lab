from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime

app = Flask(__name__)

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_url(self, url):
        """Scrape a single URL and extract structured data"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract data
            data = {
                'url': url,
                'title': soup.title.string if soup.title else 'No title',
                'headings': self._extract_headings(soup),
                'paragraphs': self._extract_paragraphs(soup),
                'links': self._extract_links(soup, url),
                'images': self._extract_images(soup, url),
                'meta_description': self._get_meta_description(soup),
                'text_content': soup.get_text(separator=' ', strip=True),
                'scraped_at': datetime.now().isoformat()
            }
            
            return {'success': True, 'data': data}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_headings(self, soup):
        """Extract all headings"""
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            if h_tags:
                headings[f'h{i}'] = [tag.get_text(strip=True) for tag in h_tags]
        return headings
    
    def _extract_paragraphs(self, soup):
        """Extract paragraphs"""
        paragraphs = soup.find_all('p')
        return [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    
    def _extract_links(self, soup, base_url):
        """Extract all links"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links.append({
                'text': link.get_text(strip=True),
                'url': full_url
            })
        return links[:50]  # Limit to 50 links
    
    def _extract_images(self, soup, base_url):
        """Extract image URLs"""
        images = []
        for img in soup.find_all('img', src=True):
            src = urljoin(base_url, img['src'])
            images.append({
                'src': src,
                'alt': img.get('alt', 'No alt text')
            })
        return images[:20]  # Limit to 20 images
    
    def _get_meta_description(self, soup):
        """Extract meta description"""
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta and meta.get('content'):
            return meta['content']
        return 'No description'
    
    def search_and_scrape(self, topic):
        """Search for a topic using DuckDuckGo and scrape results"""
        try:
            # Try multiple search strategies
            results = []
            
            # Strategy 1: Use DuckDuckGo Lite
            try:
                search_url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(topic)}"
                response = requests.get(search_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'lxml')
                
                # Find result tables in DuckDuckGo Lite
                result_tables = soup.find_all('table')
                for table in result_tables:
                    links = table.find_all('a', href=True)
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        
                        # Filter valid results
                        if (href.startswith('http') and 
                            'duckduckgo.com' not in href.lower() and
                            len(text) > 15 and
                            href not in [r['url'] for r in results]):
                            
                            results.append({
                                'title': text[:150],  # Limit title length
                                'url': href
                            })
                            
                            if len(results) >= 5:
                                break
                    
                    if len(results) >= 5:
                        break
            except Exception as e:
                print(f"Lite search failed: {e}")
            
            # Strategy 2: If no results, try HTML version
            if not results:
                try:
                    search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(topic)}"
                    response = requests.get(search_url, headers=self.headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'lxml')
                    
                    # Look for result links
                    for link in soup.find_all('a', class_='result__a'):
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        
                        if href and text and len(text) > 10:
                            results.append({
                                'title': text[:150],
                                'url': href
                            })
                            
                            if len(results) >= 5:
                                break
                except Exception as e:
                    print(f"HTML search failed: {e}")
            
            # Strategy 3: Fallback - provide sample educational results
            if not results:
                # Create informative fallback results related to the topic
                results = [
                    {
                        'title': f'Search for "{topic}" on Wikipedia',
                        'url': f'https://en.wikipedia.org/wiki/Special:Search?search={requests.utils.quote(topic)}'
                    },
                    {
                        'title': f'Search for "{topic}" on GitHub',
                        'url': f'https://github.com/search?q={requests.utils.quote(topic)}'
                    },
                    {
                        'title': f'Search for "{topic}" on Stack Overflow',
                        'url': f'https://stackoverflow.com/search?q={requests.utils.quote(topic)}'
                    },
                    {
                        'title': f'Search for "{topic}" on Reddit',
                        'url': f'https://www.reddit.com/search/?q={requests.utils.quote(topic)}'
                    },
                    {
                        'title': f'Google Search for "{topic}"',
                        'url': f'https://www.google.com/search?q={requests.utils.quote(topic)}'
                    }
                ]
            
            return {'success': True, 'results': results}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}


class LLMProcessor:
    """Process scraped content with LLM-style summarization"""
    
    def summarize_content(self, scraped_data):
        """Generate a structured summary of scraped content"""
        try:
            text_content = scraped_data.get('text_content', '')
            headings = scraped_data.get('headings', {})
            paragraphs = scraped_data.get('paragraphs', [])
            
            # Create a structured summary
            summary = {
                'title': scraped_data.get('title', 'Untitled'),
                'url': scraped_data.get('url', ''),
                'key_points': self._extract_key_points(paragraphs),
                'main_topics': self._extract_topics(headings),
                'content_structure': self._analyze_structure(headings),
                'word_count': len(text_content.split()),
                'summary_text': self._generate_summary(paragraphs, headings)
            }
            
            return summary
        
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_key_points(self, paragraphs):
        """Extract key points from paragraphs"""
        key_points = []
        for i, para in enumerate(paragraphs[:5]):  # First 5 paragraphs
            if len(para) > 50:  # Only substantial paragraphs
                # Take first sentence or truncate
                sentences = para.split('.')
                key_point = sentences[0][:200] + ('...' if len(sentences[0]) > 200 else '')
                key_points.append(key_point)
        return key_points
    
    def _extract_topics(self, headings):
        """Extract main topics from headings"""
        topics = []
        for level, heading_list in headings.items():
            topics.extend(heading_list[:3])  # Top 3 per level
        return topics[:10]
    
    def _analyze_structure(self, headings):
        """Analyze content structure"""
        structure = {}
        for level, heading_list in headings.items():
            structure[level] = len(heading_list)
        return structure
    
    def _generate_summary(self, paragraphs, headings):
        """Generate a concise summary"""
        # Get first meaningful paragraph
        first_para = next((p for p in paragraphs if len(p) > 100), '')
        
        # Get main headings
        main_headings = headings.get('h1', []) + headings.get('h2', [])
        
        summary_parts = []
        
        if main_headings:
            summary_parts.append(f"Main topics include: {', '.join(main_headings[:3])}")
        
        if first_para:
            summary_parts.append(first_para[:300] + ('...' if len(first_para) > 300 else ''))
        
        return ' '.join(summary_parts)
    
    def compare_content(self, scraped_data_list):
        """Compare multiple scraped pages"""
        comparison = {
            'total_pages': len(scraped_data_list),
            'avg_word_count': 0,
            'common_topics': [],
            'unique_insights': []
        }
        
        if scraped_data_list:
            total_words = sum(len(data.get('text_content', '').split()) 
                            for data in scraped_data_list)
            comparison['avg_word_count'] = total_words // len(scraped_data_list)
        
        return comparison


# Initialize scraper and processor
scraper = WebScraper()
llm_processor = LLMProcessor()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/scrape', methods=['POST'])
def scrape():
    """API endpoint to scrape a URL"""
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'success': False, 'error': 'URL is required'})
    
    # Scrape the URL
    result = scraper.scrape_url(url)
    
    if result['success']:
        # Process with LLM
        summary = llm_processor.summarize_content(result['data'])
        result['summary'] = summary
    
    return jsonify(result)


@app.route('/api/search-scrape', methods=['POST'])
def search_scrape():
    """API endpoint to search for a topic and scrape results"""
    data = request.json
    topic = data.get('topic')
    
    if not topic:
        return jsonify({'success': False, 'error': 'Topic is required'})
    
    # Search for the topic
    search_result = scraper.search_and_scrape(topic)
    
    if search_result['success']:
        # Optionally scrape the first result
        if search_result['results']:
            first_url = search_result['results'][0]['url']
            scrape_result = scraper.scrape_url(first_url)
            
            if scrape_result['success']:
                search_result['scraped_data'] = scrape_result['data']
                search_result['summary'] = llm_processor.summarize_content(scrape_result['data'])
    
    return jsonify(search_result)


@app.route('/api/batch-scrape', methods=['POST'])
def batch_scrape():
    """API endpoint to scrape multiple URLs"""
    data = request.json
    urls = data.get('urls', [])
    
    if not urls:
        return jsonify({'success': False, 'error': 'URLs list is required'})
    
    results = []
    for url in urls[:5]:  # Limit to 5 URLs
        result = scraper.scrape_url(url)
        if result['success']:
            result['summary'] = llm_processor.summarize_content(result['data'])
        results.append(result)
        time.sleep(1)  # Be respectful with delays
    
    return jsonify({
        'success': True,
        'results': results,
        'comparison': llm_processor.compare_content([r['data'] for r in results if r['success']])
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)