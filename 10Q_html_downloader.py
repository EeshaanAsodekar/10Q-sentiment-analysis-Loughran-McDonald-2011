import os
import pandas as pd
import requests
import glob

# Load CIK data from CSV
sp500_ciks = pd.read_csv("sp500_CIKS.csv")
# sp500_ciks = pd.read_csv("subset_ciks.csv")  # Alternative file for subset
sp500_ciks['CIK'] = sp500_ciks['CIK'].astype(str).str.zfill(10)  # Zero-pad CIKs to 10 digits
cik_list = sp500_ciks['CIK'].tolist()

# Set up request headers
headers = {'User-Agent': "your-email@example.com"}  # Replace with your email

# Define the date range for 10-Q filings
start_year = 2019
end_year = 2024

# Create directory for saving 10-Q filings
download_dir = "downloaded_10Qs"
os.makedirs(download_dir, exist_ok=True)

# Fetch filings for each CIK
for cik in cik_list:
    try:
        # Fetch metadata for the CIK
        filing_metadata_url = f'https://data.sec.gov/submissions/CIK{cik}.json'
        filing_metadata = requests.get(filing_metadata_url, headers=headers)
        filing_metadata.raise_for_status()  # Raise error if request fails

        # Parse 10-Q filings from metadata
        all_filings = pd.DataFrame.from_dict(filing_metadata.json()['filings']['recent'])
        
        # Filter invalid report dates and extract year
        all_filings = all_filings[all_filings['reportDate'].str.strip() != '']
        all_filings['year'] = all_filings['reportDate'].str[:4].astype(int, errors='ignore')
        
        # Filter for 10-Q filings in the specified date range
        ten_q_filings = all_filings[
            (all_filings['form'] == '10-Q') &
            (all_filings['year'] >= start_year) &
            (all_filings['year'] <= end_year)
        ]

        print(f"CIK: {cik} - Found {len(ten_q_filings)} 10-Q filings from {start_year} to {end_year}.")
        print(ten_q_filings[['accessionNumber', 'reportDate', 'primaryDocument']])

        # Download each 10-Q filing
        for index, row in ten_q_filings.iterrows():
            try:
                # Build document URL
                accession_number = row['accessionNumber'].replace('-', '')
                report_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{row['primaryDocument']}"

                # Download the document
                response = requests.get(report_url, headers=headers)
                response.raise_for_status()

                # Save the document as HTML
                file_path = os.path.join(download_dir, f"{cik}_{row['reportDate']}_10Q.html")
                with open(file_path, 'wb') as file:
                    file.write(response.content)

                print(f"Downloaded 10-Q report for CIK {cik} on {row['reportDate']} to {file_path}")

            except Exception as e:
                print(f"Error downloading 10-Q report for CIK {cik}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for CIK {cik}: {e}")

# Create a DataFrame to track downloaded 10-Qs
def create_tracking_dataframe(sp500_ciks, download_dir):
    """
    Create a DataFrame tracking downloaded 10-Q filings.
    
    Args:
        sp500_ciks (pd.DataFrame): DataFrame with CIK and Ticker data.
        download_dir (str): Directory with downloaded 10-Q filings.
        
    Returns:
        pd.DataFrame: Tracking DataFrame.
    """
    # List all downloaded HTML files
    html_files = glob.glob(os.path.join(download_dir, '*.html'))

    tracking_data = []

    # Extract data from each file
    for html_file in html_files:
        file_name = os.path.basename(html_file)
        cik = file_name.split('_')[0]  # Extract CIK
        report_date = file_name.split('_')[1]  # Extract report date
        ticker = sp500_ciks.loc[sp500_ciks['CIK'] == cik, 'Ticker'].values[0] if cik in sp500_ciks['CIK'].values else 'Unknown'

        tracking_data.append({
            'File Name': file_name,
            'Filing Date': report_date,
            'CIK': cik,
            'Ticker': ticker
        })

    return pd.DataFrame(tracking_data, columns=['File Name', 'Filing Date', 'CIK', 'Ticker'])

# Generate tracking DataFrame and save to CSV
tracking_df = create_tracking_dataframe(sp500_ciks, download_dir)
tracking_df.to_csv('downloaded_10Q_tracking.csv', index=False)

print(tracking_df)
