import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Load the CSV file containing S&P 500 tickers and CIKs
subset_ciks = pd.read_csv('sp500_CIKS.csv')

# Extract tickers from the DataFrame
tickers = subset_ciks['Ticker'].tolist()

# Define the date range for the last 5 years
end_date = datetime.today().date()
start_date = end_date - timedelta(days=5*365)

# Initialize an empty DataFrame to store the closing prices for each ticker
closing_prices = pd.DataFrame()

# Fetch historical closing prices for each ticker
for ticker in tickers:
    try:
        # Download historical data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # If data is found, add the 'Close' prices to the DataFrame
        if not data.empty:
            closing_prices[ticker] = data['Close']
        else:
            print(f"No data found for ticker: {ticker}")
    except Exception as e:
        # Handle errors during the data fetching process
        print(f"Error fetching data for ticker {ticker}: {e}")

# Display the first few rows of the closing prices DataFrame
print(closing_prices.head())

# Save the closing prices to a CSV file
closing_prices.to_csv("sp500_closing_prices.csv")
