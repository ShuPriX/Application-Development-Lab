// Tab switching functionality
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabId).classList.add('active');
    
    // Activate corresponding button
    event.target.classList.add('active');
    
    // Clear results
    hideResults();
}

// Show/hide loading indicator
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    hideResults();
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showResults() {
    document.getElementById('results').classList.remove('hidden');
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
}

// Scrape single URL
async function scrapeUrl() {
    const urlInput = document.getElementById('url-input');
    const url = urlInput.value.trim();
    
    if (!url) {
        alert('Please enter a URL');
        return;
    }
    
    if (!isValidUrl(url)) {
        alert('Please enter a valid URL');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/api/scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        hideLoading();
        displaySingleResult(data);
    } catch (error) {
        hideLoading();
        displayError('Failed to scrape URL: ' + error.message);
    }
}

// Search and scrape
async function searchAndScrape() {
    const topicInput = document.getElementById('topic-input');
    const topic = topicInput.value.trim();
    
    if (!topic) {
        alert('Please enter a topic to search');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/api/search-scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ topic: topic })
        });
        
        const data = await response.json();
        hideLoading();
        displaySearchResults(data);
    } catch (error) {
        hideLoading();
        displayError('Failed to search and scrape: ' + error.message);
    }
}

// Batch scrape multiple URLs
async function batchScrape() {
    const batchInput = document.getElementById('batch-urls');
    const urlsText = batchInput.value.trim();
    
    if (!urlsText) {
        alert('Please enter at least one URL');
        return;
    }
    
    const urls = urlsText.split('\n')
        .map(url => url.trim())
        .filter(url => url && isValidUrl(url));
    
    if (urls.length === 0) {
        alert('Please enter valid URLs');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/api/batch-scrape', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ urls: urls })
        });
        
        const data = await response.json();
        hideLoading();
        displayBatchResults(data);
    } catch (error) {
        hideLoading();
        displayError('Failed to batch scrape: ' + error.message);
    }
}

// Display single result
function displaySingleResult(data) {
    const resultsContent = document.getElementById('results-content');
    
    if (!data.success) {
        resultsContent.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${data.error}
            </div>
        `;
        showResults();
        return;
    }
    
    const scrapedData = data.data;
    const summary = data.summary;
    
    resultsContent.innerHTML = `
        <div class="result-card">
            <div class="result-header">
                <div>
                    <h3 class="result-title">${escapeHtml(scrapedData.title)}</h3>
                    <a href="${escapeHtml(scrapedData.url)}" target="_blank" class="result-url">${escapeHtml(scrapedData.url)}</a>
                </div>
                <span class="badge badge-success">Success</span>
            </div>
            
            ${summary ? createSummarySection(summary) : ''}
            
            <div class="summary-grid">
                <div class="summary-item">
                    <h4>Statistics</h4>
                    <p><strong>Word Count:</strong> ${summary.word_count || 0}</p>
                    <p><strong>Paragraphs:</strong> ${scrapedData.paragraphs.length}</p>
                    <p><strong>Links:</strong> ${scrapedData.links.length}</p>
                    <p><strong>Images:</strong> ${scrapedData.images.length}</p>
                </div>
                
                <div class="summary-item">
                    <h4>Content Structure</h4>
                    ${createStructureList(summary.content_structure)}
                </div>
                
                <div class="summary-item">
                    <h4>Meta Description</h4>
                    <p>${escapeHtml(scrapedData.meta_description)}</p>
                </div>
            </div>
            
            ${createKeyPointsSection(summary.key_points)}
            ${createTopicsSection(summary.main_topics)}
        </div>
    `;
    
    showResults();
}

// Display search results
function displaySearchResults(data) {
    const resultsContent = document.getElementById('results-content');
    
    if (!data.success) {
        resultsContent.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${data.error}
            </div>
        `;
        showResults();
        return;
    }
    
    let html = '<div class="result-card">';
    html += '<h3>Search Results</h3>';
    html += '<div class="search-results">';
    
    data.results.forEach((result, index) => {
        html += `
            <div class="search-result-item">
                <div class="search-result-title">${index + 1}. ${escapeHtml(result.title)}</div>
                <a href="${escapeHtml(result.url)}" target="_blank" class="search-result-link">${escapeHtml(result.url)}</a>
            </div>
        `;
    });
    
    html += '</div></div>';
    
    // If first result was scraped, display it
    if (data.scraped_data && data.summary) {
        html += `
            <div class="result-card">
                <h3>First Result Analysis</h3>
                ${createSummarySection(data.summary)}
                ${createKeyPointsSection(data.summary.key_points)}
            </div>
        `;
    }
    
    resultsContent.innerHTML = html;
    showResults();
}

// Display batch results
function displayBatchResults(data) {
    const resultsContent = document.getElementById('results-content');
    
    if (!data.success) {
        resultsContent.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${data.error}
            </div>
        `;
        showResults();
        return;
    }
    
    let html = '';
    
    // Add comparison section if available
    if (data.comparison) {
        html += `
            <div class="comparison-section">
                <h3>Batch Analysis Overview</h3>
                <div class="stat-grid">
                    <div class="stat-card">
                        <span class="stat-value">${data.comparison.total_pages}</span>
                        <span class="stat-label">Pages Scraped</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">${data.comparison.avg_word_count}</span>
                        <span class="stat-label">Avg Word Count</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add individual results
    data.results.forEach((result, index) => {
        if (result.success) {
            const scrapedData = result.data;
            const summary = result.summary;
            
            html += `
                <div class="result-card">
                    <div class="result-header">
                        <div>
                            <h3 class="result-title">${index + 1}. ${escapeHtml(scrapedData.title)}</h3>
                            <a href="${escapeHtml(scrapedData.url)}" target="_blank" class="result-url">${escapeHtml(scrapedData.url)}</a>
                        </div>
                        <span class="badge badge-success">Success</span>
                    </div>
                    
                    ${summary ? createSummarySection(summary) : ''}
                    
                    <div class="summary-grid">
                        <div class="summary-item">
                            <h4>Statistics</h4>
                            <p><strong>Words:</strong> ${summary.word_count || 0}</p>
                            <p><strong>Paragraphs:</strong> ${scrapedData.paragraphs.length}</p>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="result-card">
                    <div class="result-header">
                        <h3 class="result-title">Result ${index + 1}</h3>
                        <span class="badge badge-error">Failed</span>
                    </div>
                    <div class="error-message">
                        <strong>Error:</strong> ${escapeHtml(result.error)}
                    </div>
                </div>
            `;
        }
    });
    
    resultsContent.innerHTML = html;
    showResults();
}

// Helper function to create summary section
function createSummarySection(summary) {
    if (!summary.summary_text) return '';
    
    return `
        <div class="summary-section">
            <h3>AI Summary</h3>
            <div class="summary-item">
                <p>${escapeHtml(summary.summary_text)}</p>
            </div>
        </div>
    `;
}

// Helper function to create key points section
function createKeyPointsSection(keyPoints) {
    if (!keyPoints || keyPoints.length === 0) return '';
    
    let html = '<div class="key-points"><h4 style="margin-bottom: 15px; color: var(--text); font-weight: 600;">Key Points</h4>';
    keyPoints.forEach(point => {
        html += `<div class="key-point">${escapeHtml(point)}</div>`;
    });
    html += '</div>';
    return html;
}

// Helper function to create topics section
function createTopicsSection(topics) {
    if (!topics || topics.length === 0) return '';
    
    return `
        <div class="summary-item" style="margin-top: 20px;">
            <h4>Main Topics</h4>
            <ul>
                ${topics.map(topic => `<li>${escapeHtml(topic)}</li>`).join('')}
            </ul>
        </div>
    `;
}

// Helper function to create structure list
function createStructureList(structure) {
    if (!structure) return '<p>No structure data</p>';
    
    let html = '<ul>';
    for (const [level, count] of Object.entries(structure)) {
        html += `<li><strong>${level.toUpperCase()}:</strong> ${count}</li>`;
    }
    html += '</ul>';
    return html;
}

// Display error
function displayError(message) {
    const resultsContent = document.getElementById('results-content');
    resultsContent.innerHTML = `
        <div class="error-message">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
    showResults();
}

// Validate URL
function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle Enter key in inputs
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('url-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            scrapeUrl();
        }
    });
    
    document.getElementById('topic-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchAndScrape();
        }
    });
});