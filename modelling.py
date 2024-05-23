import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from Stock import fetch_data
from Inserting_intodatabase import insert_stock_data

#loading the data from database
def load_stock_data():
    client = MongoClient('mongodb+srv://Mongo_stock:jmMUyPuOEFZQjbgn@cluster0.vveammj.mongodb.net/')
    db = client.get_database('stock_data')
    collection = db.get_collection('stock_prices')
    
    # Fetch data from MongoDB
    data = list(collection.find())
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df
# data Preprocessing
def preprocess_data(df, ticker):
    # Filter data for the specific ticker
    df_ticker = df[df['ticker'] == ticker]
    
    # Sort by date
    df_ticker = df_ticker.sort_values('date')
    
    # Use only the closing prices for predictions
    close_prices = df_ticker['close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences for LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, scaled_data, df_ticker
#Model Building
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
#predictions
def predict_future_prices(model, data, scaler, days=15):
    predictions = []
    current_data = data.copy()  # Make a copy of the data
    
    for _ in range(days):
        # Use all available data to predict the next time step
        prediction = model.predict(current_data[np.newaxis, :, :])
        predictions.append(prediction[0, 0])
        # Update current_data to include the new prediction
        current_data = np.append(current_data, prediction.reshape(1, 1), axis=0)
    
    # Inverse transform predictions to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions
def update_database(predictions, ticker, last_date):
    client = MongoClient('mongodb+srv://Mongo_stock:jmMUyPuOEFZQjbgn@cluster0.vveammj.mongodb.net/')
    db = client.get_database('stock_data')
    collection = db.get_collection('stock_prices_predicted')

    prediction_data = []

    for i, prediction in enumerate(predictions):
        date = (last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d')
        prediction_data.append({
            'ticker': ticker,
            'date': date,
            'predicted_close': float(prediction[0])
        })

    collection.insert_many(prediction_data) 
def main():
    # Fetch and insert initial stock data into MongoDB
    #insert_stock_data()
    
    # List of stock tickers to process
    tickers = ["IBM", "AAPL", "TSLA", "META"]
    
    # Load data from MongoDB
    df = load_stock_data()
    print("load stock")
    for ticker in tickers:
        # Preprocess data
        X, y, scaler, full_data, df_ticker = preprocess_data(df, ticker)
        print("precprocess")
        # Build and train LSTM model
        model = build_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=20, batch_size=32)
        print("model built")
        # Get the last available date for the current ticker
        last_date_str = df_ticker['date'].max()
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
        
        # Predict future prices using the entire historical data
        predictions = predict_future_prices(model, full_data, scaler)
        
        # Update the database with predictions
        update_database(predictions, ticker, last_date)

if __name__ == "__main__":
    main()  
