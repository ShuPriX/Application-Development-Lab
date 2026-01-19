// Managing Tabs
function switchTab(tabId) {
    document.querySelectorAll('.card').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    
    // Highlight button
    const btns = document.querySelectorAll('.nav-btn');
    if(tabId === 'classification') btns[0].classList.add('active');
    else btns[1].classList.add('active');
}

// --- CLASSIFICATION LOGIC ---
document.getElementById('imageInput').addEventListener('change', function(e) {
    if(e.target.files[0]) {
        const reader = new FileReader();
        reader.onload = function(evt) {
            document.getElementById('preview').src = evt.target.result;
            document.getElementById('result-container').classList.remove('hidden');
            document.getElementById('prediction-text').innerText = "Ready to analyze";
            document.getElementById('confidence-text').innerText = "-";
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});

async function processImage() {
    const file = document.getElementById('imageInput').files[0];
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const btn = document.querySelector('#classification .action-btn');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.disabled = true;
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('http://127.0.0.1:5000/classify', { method: 'POST', body: formData });
        const data = await res.json();
        
        if(data.error) {
            alert(data.error);
        } else {
            document.getElementById('prediction-text').innerText = data.label;
            document.getElementById('confidence-text').innerText = data.confidence;
        }
    } catch (e) {
        alert("Error connecting to backend.");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

// --- STOCK LOGIC ---
let stockChart;

async function predictStock() {
    const tickerInput = document.getElementById('tickerInput');
    const ticker = tickerInput.value.trim();
    const btn = document.getElementById('predictBtn');
    
    if (!ticker) {
        alert("Please enter a ticker symbol (e.g., AAPL)");
        return;
    }
    
    // UI Feedback
    btn.innerText = "Fetching Data... (Wait 5-10s)";
    btn.disabled = true;
    btn.style.opacity = "0.7";

    try {
        console.log("Sending request for:", ticker);
        
        const res = await fetch('http://127.0.0.1:5000/predict_stock', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker: ticker})
        });

        const data = await res.json();
        
        if (data.error) {
            alert("Backend Error: " + data.error);
        } else {
            renderChart(data);
        }

    } catch(e) {
        console.error("Fetch Error:", e);
        alert("Failed to connect to backend. Is python app.py running?");
    } finally {
        btn.innerText = "Generate Forecast";
        btn.disabled = false;
        btn.style.opacity = "1";
    }
}

function renderChart(data) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    if (stockChart) {
        stockChart.destroy();
    }

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Historical Data',
                    data: data.actual,
                    borderColor: '#b2bec3',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Trend Forecast (5 Years)',
                    data: data.linear,
                    borderColor: '#2d3436', // Dark Grey
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'LSTM Validation',
                    data: data.lstm,
                    borderColor: '#00b894', // Green
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: { position: 'bottom' },
                title: { display: true, text: 'Stock Prediction Analysis' }
            },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
                y: { grid: { borderDash: [5, 5] } }
            }
        }
    });
}