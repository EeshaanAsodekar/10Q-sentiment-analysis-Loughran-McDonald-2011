import pandas as pd
import requests

# Load the S&P 500 composition from an Excel file
file_path = "sp500_composition.xlsx"  # Specify the path to the Excel file
sp500_df = pd.read_excel(file_path)

# Extract unique tickers from the dataframe and ensure they are uppercase
tickers = sp500_df['Ticker'].dropna().unique().tolist()
tickers = [ticker.upper() for ticker in tickers]

# Set your User-Agent header for the requests
headers = {'User-Agent': 'Your Name (your_email@example.com)'}  # Replace with your name and email

# Fetch the ticker to CIK mapping from the SEC's website
url = "https://www.sec.gov/files/company_tickers.json"
response = requests.get(url, headers=headers)
data = response.json()

# Create a dictionary to map tickers to CIKs, ensuring CIKs are zero-padded to 10 digits
ticker_cik_map = {}
for item in data.values():
    ticker = item['ticker'].upper()
    cik = str(item['cik_str']).zfill(10)
    ticker_cik_map[ticker] = cik

# Map S&P 500 tickers to their corresponding CIKs
cik_dict = {}
for ticker in tickers:
    if ticker in ticker_cik_map:
        cik_dict[ticker] = ticker_cik_map[ticker]
    else:
        print(f"CIK not found for ticker: {ticker}")

# Convert the mapping dictionary to a DataFrame
cik_df = pd.DataFrame(list(cik_dict.items()), columns=['Ticker', 'CIK'])

# Display the resulting DataFrame
print(cik_df)

# Save the ticker to CIK mapping to a CSV file
cik_df.to_csv("sp500_CIKS.csv", index=False)
