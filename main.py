import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from textblob import TextBlob

# Function to get stock price
def get_stock_price(ticker):
    return yf.Ticker(ticker).history(period='1d').iloc[-1].Close

# Function to calculate simple moving average (SMA)
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Function to calculate exponential moving average (EMA)
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Function to calculate relative strength index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to perform sentiment analysis on stock-related tweets
def analyze_sentiment(ticker):
    # Placeholder for demo purposes; replace with actual sentiment analysis logic
    tweets = ["Positive tweet about " + ticker, "Neutral tweet", "Negative tweet"]
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    average_sentiment = sum(sentiments) / len(sentiments)
    return average_sentiment

# Streamlit App
st.title('Advanced Stock Market Analysis App')

# User Input
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')

# Historical Data
historical_data = yf.Ticker(ticker).history(period='1y')

# Display Stock Price
current_price = get_stock_price(ticker)
st.write(f"Current Stock Price ({ticker}): ${current_price}")

# Display Historical Data
st.subheader(f'Historical Data ({ticker})')
st.dataframe(historical_data.tail(10))

# Technical Indicators
st.subheader('Technical Indicators')
sma_window = st.slider('Select SMA Window:', min_value=1, max_value=50, value=10)
historical_data['SMA'] = calculate_sma(historical_data['Close'], sma_window)
st.line_chart(historical_data[['Close', 'SMA']])

ema_window = st.slider('Select EMA Window:', min_value=1, max_value=50, value=10)
historical_data['EMA'] = calculate_ema(historical_data['Close'], ema_window)
st.line_chart(historical_data[['Close', 'EMA']])

rsi_window = st.slider('Select RSI Window:', min_value=1, max_value=50, value=14)
historical_data['RSI'] = calculate_rsi(historical_data['Close'], rsi_window)
st.line_chart(historical_data['RSI'])

# Sentiment Analysis
st.subheader('Sentiment Analysis')
sentiment_score = analyze_sentiment(ticker)
st.write(f"Average Sentiment Score for {ticker}: {sentiment_score:.2f}")

# Conclusion and Recommendations (Placeholder)
st.subheader('Conclusion and Recommendations')
st.write("Based on the analysis, it is recommended to...")
