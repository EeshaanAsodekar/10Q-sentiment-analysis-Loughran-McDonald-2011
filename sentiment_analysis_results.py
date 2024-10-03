import os
import pandas as pd
from collections import Counter
import warnings
import math

### Loughran-McDonald approach
# Load negative words from the CSV file
words_df = pd.read_csv('NEGLoughranMcDonald_MasterDictionary_2018NEG.csv')
words_list = words_df.iloc[:, 0].fillna('').astype(str).tolist()  # Ensure words are strings

# Initialize word counts dictionary
word_counts = {word: [] for word in words_list}

# Directory containing text files
# folder_path = 'sec_parser_10Q_txt'
folder_path = 'sec_parser_subset'

# Get list of .txt files
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Count occurrences of each word in each file
for txt_file in txt_files:
    with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Convert to lowercase

    word_counter = Counter(text.split())

    # Append word counts to the dictionary
    for word in words_list:
        word_counts[word].append(word_counter.get(word.lower(), 0))

# Convert word counts to DataFrame
count_matrix_df = pd.DataFrame(word_counts, index=txt_files)

# Save word count matrix to CSV
count_matrix_df.to_csv('word_count_matrix.csv')

# Load additional document statistics
# stats_df = pd.read_csv('Parser_motherload.csv')
stats_df = pd.read_csv('Parser.csv')
stats_df.head()

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Calculate TF-IDF
df_tf_idf = count_matrix_df.copy()
df_i = [0] * count_matrix_df.shape[1]

for i in range(df_tf_idf.shape[0]):  # Iterate over documents
    for j in range(df_tf_idf.shape[1]):  # Iterate over words
        if df_tf_idf.iloc[i, j]:
            df_tf_idf.iloc[i, j] = (1 + math.log(df_tf_idf.iloc[i, j])) / (1 + math.log(stats_df['number of words,'][i]))
            df_i[j] += 1

df_i = [math.log(df_tf_idf.shape[0] / i) if i else 0 for i in df_i]
df_tf_idf = df_tf_idf.mul(df_i, axis=1)

print(df_tf_idf.head(5))
print(df_tf_idf.shape)

### Weighting
# Multiply word counts by weights (TF-IDF)
weighted_counts = df_tf_idf * count_matrix_df

# Sum weighted counts for each document
weighted_sum = weighted_counts.sum(axis=1)

# Create a DataFrame for the results
weighted_sum_df = pd.DataFrame(weighted_sum, columns=['Weighted_Negative_Sum'])

# Display the result
print(weighted_sum_df.head())





### Harvard Dictionary approach

# Load negative words from the text file
words_df = pd.read_csv('Harvard_Negative_Words_List.txt', header=None)
words_list = words_df.iloc[:, 0].fillna('').astype(str).tolist()  # Ensure words are valid strings

# Initialize dictionary for word counts
word_counts = {word: [] for word in words_list}

# Directory containing text files
# folder_path = 'sec_parser_10Q_txt'
folder_path = 'sec_parser_subset'

# Get list of text files in the folder
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Count occurrences of each word in the text files
for txt_file in txt_files:
    with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as file:
        text = file.read().lower()  # Convert to lowercase for case-insensitive matching

    word_counter = Counter(text.split())

    # Append word counts to the dictionary
    for word in words_list:
        word_counts[word].append(word_counter.get(word.lower(), 0))

# Convert word counts to DataFrame
count_matrix_df = pd.DataFrame(word_counts, index=txt_files)

# Save word count matrix to CSV
count_matrix_df.to_csv('word_count_matrix.csv')

# Load document statistics
# stats_df = pd.read_csv('Parser_motherload.csv')
stats_df = pd.read_csv('Parser.csv')
stats_df.head()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings

# Calculate TF-IDF
df_tf_idf = count_matrix_df.copy()
df_i = [0] * count_matrix_df.shape[1]

for i in range(df_tf_idf.shape[0]):  # Iterate over documents
    for j in range(df_tf_idf.shape[1]):  # Iterate over words
        if df_tf_idf.iloc[i, j]:
            df_tf_idf.iloc[i, j] = (1 + math.log(df_tf_idf.iloc[i, j])) / (1 + math.log(stats_df['number of words,'][i]))
            df_i[j] += 1

df_i = [math.log(df_tf_idf.shape[0] / i) if i else 0 for i in df_i]
df_tf_idf = df_tf_idf.mul(df_i, axis=1)
df_ = df_tf_idf.copy()

print(df_.head(5))
print(df_.shape)

### Weighting
# Apply element-wise multiplication of weights and counts
weighted_counts = df_ * count_matrix_df

# Sum the weighted counts to get total weighted negative words per document
weighted_sum = weighted_counts.sum(axis=1)

# Create DataFrame to store the result
har_weighted_sum_df = pd.DataFrame(weighted_sum, columns=['Weighted_Negative_Sum'])

# Display the result
print(har_weighted_sum_df.head())





### Preparing DataFrame for excess returns

# Function to extract CIK and date from the index string
def extract_cik_and_date(index):
    parts = index.split('_')  # Split the index into CIK and date
    cik = parts[0]            # Extract CIK (first part)
    date = parts[1]           # Extract date (second part)
    return cik, date

# Apply the function to extract CIK and Release Date into separate columns
weighted_sum_df['CIK'] = weighted_sum_df.index.map(lambda x: extract_cik_and_date(x)[0])
weighted_sum_df['Release_Date'] = weighted_sum_df.index.map(lambda x: extract_cik_and_date(x)[1])

# Display the DataFrame with new CIK and Release Date columns
print(weighted_sum_df.head())

### Computing excess returns

import pandas as pd

# Load SP500 and stock prices data
file_path = 'sp500_closing_prices.csv'  # Update the path if necessary
sp500_closing_prices = pd.read_csv(file_path)

# Set 'Date' as the DataFrame index
sp500_closing_prices.set_index('Date', inplace=True)

# Convert all columns to numeric, setting invalid parsing results to NaN
sp500_closing_prices = sp500_closing_prices.apply(pd.to_numeric, errors='coerce')

# Calculate daily returns for all stocks, including SP500
returns = sp500_closing_prices.pct_change()

# Calculate excess returns (stock returns minus SP500 index returns)
excess_returns = returns.subtract(returns['^GSPC'], axis=0)

# Save excess returns to a CSV file
excess_returns.to_csv("excess_returns.csv")

# Display the first few rows of excess returns
print(excess_returns.head())





### Renaming columns from tickers to CIKs

# Load excess returns data and SP500 CIKs mapping
excess_returns_file_path = 'excess_returns.csv'  # Path to excess returns data
sp500_ciks_file_path = 'sp500_ciks.csv'  # Path to SP500 CIKs mapping file

# Load excess returns and SP500 CIKs DataFrames
excess_returns = pd.read_csv(excess_returns_file_path)  # Adjust this if already loaded
sp500_ciks_df = pd.read_csv(sp500_ciks_file_path)

# Set 'Date' as the index for the excess_returns DataFrame
excess_returns.set_index('Date', inplace=True)

# Ensure CIK column is treated as a string with leading zeros
sp500_ciks_df['CIK'] = sp500_ciks_df['CIK'].astype(str).str.zfill(10)

# Create a dictionary to map Ticker to CIK
ticker_to_cik = dict(zip(sp500_ciks_df['Ticker'], sp500_ciks_df['CIK']))

# Rename columns in excess_returns using CIKs, keep original names for unmatched tickers
new_columns = [ticker_to_cik.get(col, col) for col in excess_returns.columns]
excess_returns.columns = new_columns

# Save the updated excess returns data (optional)
excess_returns.to_csv('updated_excess_returns.csv', index=False)

# Display the first few rows of the updated excess returns DataFrame
print(excess_returns.head())

# Extract CIK and Release Date for weighted_sum_df
weighted_sum_df['CIK'] = weighted_sum_df.index.map(lambda x: extract_cik_and_date(x)[0])
weighted_sum_df['Release_Date'] = weighted_sum_df.index.map(lambda x: extract_cik_and_date(x)[1])

# Extract CIK and Release Date for har_weighted_sum_df
har_weighted_sum_df['CIK'] = har_weighted_sum_df.index.map(lambda x: extract_cik_and_date(x)[0])
har_weighted_sum_df['Release_Date'] = har_weighted_sum_df.index.map(lambda x: extract_cik_and_date(x)[1])





### Plotting results: contrasting the two approaches
import pandas as pd
import matplotlib.pyplot as plt

# Compute 4-day excess returns for both weighted_sum_df and har_weighted_sum_df
for df in [weighted_sum_df, har_weighted_sum_df]:
    for idx, row in df.iterrows():
        cik = row['CIK']
        release_date = row['Release_Date']

        try:
            # Find the closest date in excess_returns matching the release date
            closest_idx = excess_returns.index.searchsorted(release_date)
            if closest_idx < len(excess_returns):
                closest_date = excess_returns.index[closest_idx]
                # Sum excess returns for the next 4 days (up to 5 rows)
                excess_return_4_days = excess_returns.loc[closest_date:closest_date].iloc[:5][cik].sum()

                # Store the 4-day excess return if valid, otherwise set to 0
                df.at[idx, '4_day_excess_return'] = excess_return_4_days if isinstance(excess_return_4_days, (int, float)) else 0
        except (KeyError, IndexError):
            # Handle missing CIK or date issues by setting excess return to 0
            df.at[idx, '4_day_excess_return'] = 0

# Remove rows with zero 4-day excess return for both datasets
non_zero_weighted_df = weighted_sum_df[weighted_sum_df['4_day_excess_return'] != 0]
non_zero_har_weighted_df = har_weighted_sum_df[har_weighted_sum_df['4_day_excess_return'] != 0]

# Create quintiles for 'Weighted_Negative_Sum' and label them accordingly
non_zero_weighted_df['negative_quintile'] = pd.qcut(non_zero_weighted_df['Weighted_Negative_Sum'], 5, labels=["Low", "2", "3", "4", "High"])
non_zero_har_weighted_df['negative_quintile'] = pd.qcut(non_zero_har_weighted_df['Weighted_Negative_Sum'], 5, labels=["Low", "2", "3", "4", "High"])

# Calculate the median 4-day excess return for each quintile
median_excess_return_by_quintile_weighted = non_zero_weighted_df.groupby('negative_quintile')['4_day_excess_return'].median()
median_excess_return_by_quintile_har_weighted = non_zero_har_weighted_df.groupby('negative_quintile')['4_day_excess_return'].median()

# Plot median 4-day excess return for each quintile
plt.figure(figsize=(8, 6))

# Plot for Loughran-McDonald
plt.plot(median_excess_return_by_quintile_weighted.index, median_excess_return_by_quintile_weighted.values, 
         marker='o', linestyle='-', color='blue', label='Loughran-McDonald')

# Plot for Harvard dictionary
plt.plot(median_excess_return_by_quintile_har_weighted.index, median_excess_return_by_quintile_har_weighted.values, 
         marker='o', linestyle='-', color='green', label='Harvard')

# Set axis labels and title
plt.xlabel('Quintile (based on Weighted Negative Sum)')
plt.ylabel('Median 4-Day Excess Return')
plt.title('Median 4-Day Excess Return Across Weighted Negative Sum Quintiles')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)

# Adjust y-axis limits for better visualization
y_min = min(median_excess_return_by_quintile_weighted.min(), median_excess_return_by_quintile_har_weighted.min()) - 0.001
y_max = max(median_excess_return_by_quintile_weighted.max(), median_excess_return_by_quintile_har_weighted.max()) + 0.001
plt.ylim(y_min, y_max)

# Increase precision on y-axis tick labels
plt.yticks([round(x, 4) for x in plt.yticks()[0]])

# Display the plot
plt.show()
plt.savefig("LoughranMcDonald_vs_Harvard.png")