# Sentiment Analysis of 10-Q Filings Using Custom Financial Dictionaries

This project implements a textual analysis framework to evaluate the tone of corporate 10-Q filings, focusing on sentiment. The analysis leverages the Loughran-McDonald (2011) financial word list and compares it with the Harvard-IV dictionary, highlighting the differences in sentiment classification between financial and general-purpose word lists.

## Objective
The goal of this project is to assess how the tone of 10-Q filings correlates with excess stock returns around the filing date, using both financial and general negative word lists. This involves:
- Textual analysis of 10-Q filings using custom financial sentiment dictionaries.
- Calculation of 4-day excess stock returns following 10-Q filings.
- Comparison of results obtained from the Loughran-McDonald and Harvard word lists.

## Data
The project processes S&P 500 constituent data and 10-Q filings downloaded from the SEC. It utilizes historical stock price data from Yahoo Finance to compute excess returns.

### Input Files:
1. **sp500_composition.csv**: S&P 500 constitutent firms and tickers

## Key Components
1. **Downloading & Extracting 10Q Filings**: Downloading the 10Q filings for all the S&P500 constituents
2. **Downloading and Computing Excess Returns**: Using yfinance to download the closing prices of the S&P500 constituents
3. **Negative Scoring/Weighting for each 10Q**: A negative weight is computed for each 10Q based on the count of negative words (dictionary dependent) 
4. **Negative Weight (Sentiment) vs. 4-day Excess Return**: Analyze the relationship between sentiment scores and excess returns, grouped into quintiles based on negative sentiment scores.

## Usage
1. Run `10Q_html_downloader.py` to download all the 10Qs of the S&P500 constitutents 
2. Run `10Q_extractor.py` to extract the text from the downloaded 10Qs
3. Run `Generic_Parser.py` to get word count and other intermediate computations; returns a `Parser.csv` file.
4. Run `get_sp500_closing_data.py` to get S&P500 closing prices
5. Run `get_sp500_cik.py` to get S&P500 CIK
6. Run `sentiment_analysis_results.py` to compute the 10Q weights and visualize the trend of negative sentiment v/s 4 day cumulative returns

## Dependencies
Install the env: `SentimentAnalysis_LoughranMcDonald_env.yml` 

## Results
The project demonstrates that financial-specific word lists, such as the Loughran-McDonald list, provide a more accurate measure of negative sentiment in 10-Q filings than general-purpose dictionaries like Harvard-IV. The analysis also shows a significant relationship between the negative tone of a 10-Q and the corresponding 4-day stock price reaction.
![Loughran-McDonald vs. Harvard Results](https://github.com/EeshaanAsodekar/10Q-sentiment-analysis-Loughran-McDonald-2011/blob/main/LoughranMcDonald_vs_Harvard.png)

## References
Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10‐Ks. *The Journal of Finance*, 66(1), 35-65.