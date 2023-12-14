import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from textblob import TextBlob
import numpy as np

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
    tweets = ["Positive tweet about " + ticker, "Neutral tweet", "Negative tweet"]
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    average_sentiment = sum(sentiments) / len(sentiments)
    return average_sentiment

# Function to recommend investment
def recommend_investment(rsi, sentiment_score):
    if rsi < 30 and sentiment_score > 0:
        return "Strong Buy"
    elif rsi < 50 and sentiment_score > 0.5:
        return "Buy"
    elif rsi > 70 and sentiment_score < 0:
        return "Strong Sell"
    elif rsi > 50 and sentiment_score < -0.5:
        return "Sell"
    else:
        return "Hold"

# Function to suggest investment amount
def suggest_investment_amount(current_price, risk_percentage=5):
    risk_amount = current_price * (risk_percentage / 100)
    return risk_amount

# Function to calculate win and loss probability
def calculate_win_loss_probability(sentiment_score):
    win_probability = (sentiment_score + 1) / 2
    loss_probability = 1 - win_probability
    return win_probability, loss_probability

# Streamlit App
st.title('Advanced Stock Market Analysis App')

# User Input
selected_ticker = st.selectbox('Select Stock Ticker:', ['AAPL', 'META', 'GOOGL', 'MSFT', 'X', 'IBM'])

# Historical Data
historical_data = yf.Ticker(selected_ticker).history(period='1y')

# Display Stock Price
current_price = get_stock_price(selected_ticker)
st.write(f"Current Stock Price ({selected_ticker}): ${current_price}")

# Display Historical Data
st.subheader(f'Historical Data ({selected_ticker})')
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
sentiment_score = analyze_sentiment(selected_ticker)
st.write(f"Average Sentiment Score for {selected_ticker}: {sentiment_score:.2f}")

# Investment Recommendations
st.subheader('Investment Recommendations')
investment_recommendation = recommend_investment(historical_data['RSI'].iloc[-1], sentiment_score)
st.write(f"Recommendation: {investment_recommendation}")

# Suggested Investment Amount
st.subheader('Suggested Investment Amount')
risk_percentage = st.slider('Select Risk Percentage:', min_value=1, max_value=10, value=5)
investment_amount = suggest_investment_amount(current_price, risk_percentage)
st.write(f"Suggested Investment Amount: ${investment_amount:.2f}")

# Win and Loss Probability
st.subheader('Win and Loss Probability')
win_probability, loss_probability = calculate_win_loss_probability(sentiment_score)
st.write(f"Win Probability: {win_probability:.2%}")
st.write(f"Loss Probability: {loss_probability:.2%}")

