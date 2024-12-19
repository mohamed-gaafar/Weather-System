from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Load dataset and model
data = pd.read_csv('Cairo,Egypt 2023-01-01 to 2024-06-25.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data[['temp']]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Load the trained LSTM model
try:
    model = load_model('lstm_temperature_forecasting_model.h5', compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to create dataset for LSTM forecasting
def create_forecast(window_size):
    last_window = data_scaled[-(window_size + 7): -7]
    actual_values = data_scaled[-7:]
    last_window = np.expand_dims(last_window, axis=0)

    forecasted_temps = []
    for _ in range(7):
        predicted_temp_scaled = model.predict(last_window)
        forecasted_temps.append(predicted_temp_scaled[0, 0])
        last_window = np.roll(last_window, -1, axis=1)
        last_window[:, -1, 0] = predicted_temp_scaled

    forecasted_temps = scaler.inverse_transform(np.array(forecasted_temps).reshape(-1, 1))
    actual_temps = scaler.inverse_transform(actual_values)

    return forecasted_temps.flatten().tolist(), actual_temps.flatten().tolist()

@app.route('/forecast', methods=['GET'])
def forecast():
    window_size = int(request.args.get('window_size', 40))
    forecasted_temps, actual_temps = create_forecast(window_size)
    response = {
        'forecasted_temps': forecasted_temps,
        'actual_temps': actual_temps
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
