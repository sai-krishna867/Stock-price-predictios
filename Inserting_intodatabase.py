from pymongo import MongoClient
from Stock import fetch_data

def insert_stock_data():
    stock_data_list = fetch_data()
    #print(stock_data_list)
    # Connect to MongoDB
    # Replace 'your_connection_uri' with your actual MongoDB connection URI
    client = MongoClient('mongodb+srv://Mongo_stock:jmMUyPuOEFZQjbgn@cluster0.vveammj.mongodb.net/')
    
    # Access or create the database and collection
    db = client.get_database('stock_data')  # Database name
    collection = db.get_collection('stock_prices')  # Collection name

    # Insert the collected data into MongoDB
    collection.insert_many(stock_data_list)
insert_stock_data()