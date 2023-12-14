import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

# Function to get stock price
def get_stock_price(ticker):
    return yf.Ticker(ticker).history(period='1d').iloc[-1].Close

# Function to calculate simple moving average
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Function to calculate exponential moving average
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Function to calculate relative strength index (RSI)
def calculate_rsi(data, window):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to analyze sentiment
def analyze_sentiment(ticker):
    return 0.2  

# Function to recommend investment
def recommend_investment(rsi, sentiment_score):
    if rsi < 30 and sentiment_score > 0:
        return "Buy"
    elif rsi > 70 and sentiment_score < 0:
        return "Sell"
    else:
        return "Hold"

# Function to suggest investment amount
def suggest_investment_amount(current_price, risk_percentage):
    return current_price * risk_percentage / 100

# Function to calculate win and loss probability
def calculate_win_loss_probability(sentiment_score):
    win_probability = max(0, min(1, (sentiment_score + 1) / 2))
    loss_probability = 1 - win_probability
    return win_probability, loss_probability

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(data, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe_ratio

# Function to calculate Sortino ratio
def calculate_sortino_ratio(data, risk_free_rate=0.02):
    returns = data.pct_change().dropna()
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() - risk_free_rate) / downside_returns.std()
    return sortino_ratio

# Function to calculate Maximum Drawdown
def calculate_max_drawdown(data):
    cum_returns = (1 + data.pct_change()).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

# Function to suggest investment based on financial condition
def suggest_investment_based_on_financial_condition(financial_condition):
    if financial_condition == "Excellent":
        return "Based on your excellent financial condition, we suggest investing in well-established companies with a low-risk profile."
    elif financial_condition == "Moderate":
        return "Based on your moderate financial condition, consider a diversified portfolio with a mix of growth and value stocks."
    elif financial_condition == "Poor":
        return "Considering your financial condition, focus on risk management and consider conservative investment strategies."

# Streamlit App
st.set_page_config(
    page_title="Stock Analysis App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Background Color
bg_color = "#2C3E50"

# Set Page Style
st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title('Advanced Stock Market Analysis App')

# User Input
selected_ticker = st.selectbox('Select Stock Ticker:', ['AAPL', 'META', 'GOOGL', 'MSFT', 'X', 'IBM', 'AMZN', 'TSLA', 'GOOGL', 'NFLX', 'BABA', 'JNJ', 'PG', 'JPM', 'GS', 'DIS', 'CSCO', 'GE'])

# Historical Data
historical_data = yf.Ticker(selected_ticker).history(period='5y')  

# Display Stock Price
current_price = get_stock_price(selected_ticker)
st.write(f"Current Stock Price ({selected_ticker}): ${current_price}")

# Display Data Duration
st.write(f"Data Duration: {historical_data.index[0].strftime('%Y-%m-%d')} to {historical_data.index[-1].strftime('%Y-%m-%d')}")

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

# Performance Metrics
st.subheader('Performance Metrics')
sharpe_ratio = calculate_sharpe_ratio(historical_data['Close'])
sortino_ratio = calculate_sortino_ratio(historical_data['Close'])
max_drawdown = calculate_max_drawdown(historical_data['Close'])
st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")
st.write(f"Sortino Ratio: {sortino_ratio:.4f}")
st.write(f"Maximum Drawdown: {max_drawdown:.2%}")

# Financial Condition Input
st.subheader('Financial Condition Input')
financial_condition = st.selectbox('Select Your Financial Condition:', ['Excellent', 'Moderate', 'Poor'])

# Suggest Investment Based on Financial Condition
investment_suggestion = suggest_investment_based_on_financial_condition(financial_condition)
st.write(f"Investment Suggestion: {investment_suggestion}")
