import numpy as np
import yfinance as yf
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
from PIL import Image
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# --- CLASSIFICATION MODEL ---
# Load once at startup
print("Loading MobileNetV2...")
model_cnn = MobileNetV2(weights='imagenet')
print("Model loaded.")

def prepare_image(img_bytes):
    # Convert to RGB to handle PNGs with transparency
    img = Image.open(BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        processed_img = prepare_image(file.read())
        
        preds = model_cnn.predict(processed_img)
        results = decode_predictions(preds, top=1)[0]
        
        return jsonify({
            'label': results[0][1],
            'confidence': f"{results[0][2]*100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- STOCK PREDICTION ---
@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL').upper()
        
        # Fetch Data
        stock = yf.Ticker(ticker).history(period='5y', interval='1d')
        
        if stock.empty or len(stock) < 60:
            return jsonify({'error': 'Invalid Ticker or not enough data'}), 400

        # Extract Close prices
        values = stock[['Close']].values 
        
        # --- 1. PREPARE DATES ---
        future_days = 1250 # 5 years
        last_date = stock.index[-1]
        
        # Generate future dates
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
        historical_dates = stock.index.strftime('%Y-%m-%d').tolist()
        all_dates = historical_dates + future_dates_str

        # --- 2. LINEAR REGRESSION (Trend) ---
        current_len = len(values)
        X_history = np.arange(current_len).reshape(-1, 1)
        X_future = np.arange(current_len + future_days).reshape(-1, 1)
        
        lr = LinearRegression()
        lr.fit(X_history, values)
        lr_projection = lr.predict(X_future).flatten()

        # --- 3. LSTM (Validation) ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(values)
        
        look_back = 60
        X_lstm, y_lstm = [], []
        
        for i in range(look_back, len(scaled_data)):
            X_lstm.append(scaled_data[i-look_back:i, 0])
            y_lstm.append(scaled_data[i, 0])
            
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        
        # Train simple LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_lstm, y_lstm, batch_size=32, epochs=1, verbose=0)
        
        # Predict
        lstm_preds = model.predict(X_lstm)
        lstm_preds = scaler.inverse_transform(lstm_preds).flatten()

        # --- 4. FORMAT RESPONSE ---
        # Align data for chart.js
        # Actual: History + Nulls
        actual_data = values.flatten().tolist() + [None] * future_days
        
        # LSTM: Nulls (for lookback) + Preds + Nulls (for future)
        lstm_padding = [None] * look_back
        lstm_future_pad = [None] * future_days
        lstm_chart_data = lstm_padding + lstm_preds.tolist() + lstm_future_pad

        return jsonify({
            'dates': all_dates,
            'actual': actual_data,
            'linear': lr_projection.tolist(),
            'lstm': lstm_chart_data
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)