def fetch_data():
    from datetime import datetime, timedelta
    import requests
    import time
    from pymongo import MongoClient, ASCENDING
    stocks =["IBM","AAPL", "TSLA", "META"]
    base_url = "https://api.polygon.io/v2/aggs/ticker/"
    api_key ="Ajw0EeaGpWTPKjDJCqIxFmbqTxGgXVB8"
    stock_list=[]
    x = '2023-01-09'
    y= datetime.now() - timedelta(1)
    y = y.strftime('%Y-%m-%d')
    stock_data_list = []
    for i in stocks:
        url = f"{base_url}{i}/range/1/day/{x}/{y}?adjusted=true&sort=asc&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()
        if 'results' in data:
                for result in data['results']:
                    stock_info = {
                        'ticker': i,
                        'date': datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                        'open': result['o'],
                        'high': result['h'],
                        'low': result['l'],
                        'close': result['c'],
                        'volume': result['v']
                    }
                    stock_data_list.append(stock_info)
        time.sleep(20)
    return stock_data_list
      
        