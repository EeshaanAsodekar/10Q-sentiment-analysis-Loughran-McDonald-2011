# Sentiment Analysis of 10-K Filings Using Custom Financial Dictionaries

This project implements a textual analysis framework to evaluate the tone of corporate 10-K filings, focusing on financial sentiment. The analysis leverages the Loughran-McDonald (2011) financial word list and compares it with the Harvard-IV dictionary, highlighting the differences in sentiment classification between financial and general-purpose word lists.

## Objective
The goal of this project is to assess how the tone of 10-K filings correlates with excess stock returns around the filing date, using both financial and general negative word lists. This involves:
- Textual analysis of 10-K filings using custom financial sentiment dictionaries.
- Calculation of 4-day excess stock returns following 10-K filings.
- Comparison of results obtained from the Loughran-McDonald and Harvard word lists.

## Data
The project processes S&P 500 constituent data and 10-K filings downloaded from the SEC. It utilizes historical stock price data from Yahoo Finance to compute excess returns.

### Input Files:
1. **sp500_CIKS.csv**: Contains CIK and Ticker mappings for S&P 500 companies.
2. **NEGLoughranMcDonald_MasterDictionary_2018NEG.csv**: The Loughran-McDonald financial sentiment dictionary.
3. **Harvard_Negative_Words_List.txt**: The Harvard-IV dictionary for sentiment analysis.
4. **sp500_closing_prices.csv**: Historical stock prices for S&P 500 companies.
5. **10-Q HTML files**: 10-K filings downloaded from the SEC.

## Key Components
1. **Word Count Analysis**: Parses 10-K filings and counts occurrences of sentiment-related words from both the Loughran-McDonald and Harvard word lists.
3. **Tone vs. Return Correlation**: Analyzes the correlation between sentiment scores and excess returns, grouped into quintiles based on negative sentiment scores.

## Usage
1. Download the required S&P 500 data and word lists.
2. Run the `textual_analysis.py` script to perform the word count analysis on 10-K filings.
3. Run the `excess_returns_analysis.py` script to calculate the 4-day excess returns for each company.
4. Visualize the results using the provided plotting functions to compare sentiment analysis results from different word lists.

## Dependencies
Env used: SentimentAnalysis_LoughranMcDonald_env.yml 

## Results
The project demonstrates that financial-specific word lists, such as the Loughran-McDonald list, provide a more accurate measure of negative sentiment in 10-K filings than general-purpose dictionaries like Harvard-IV. The analysis also shows a significant relationship between the negative tone of a 10-K and the corresponding 4-day stock price reaction.

## References
Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10‐Ks. *The Journal of Finance*, 66(1), 35-65.