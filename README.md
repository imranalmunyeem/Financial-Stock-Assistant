# Project Name: Financial Stick Assistant
### Description
        The Advanced Stock Market Analysis App is a comprehensive tool designed to empower users with advanced insights into stock market data for informed investment decisions.
        
### Features: 
        Technical Indicators: SMA, EMA, RSI. Sentiment Analysis: Average sentiment score. 
        Investment Recommendations: Buy, Sell, Hold. 
        Suggested Investment Amount: Customizable based on risk percentage. 
        Win and Loss Probability: Probabilities derived from sentiment analysis. 
        Performance Metrics: Sharpe ratio, Sortino ratio, Maximum Drawdown. 
        Financial Condition Input: User's financial condition selection. 
        Investment Suggestion: Tailored investment suggestions based on financial condition.

### Tools and Packages Used
        Python: Programming language for backend development. 
        Streamlit: Framework for building interactive web applications. 
        yfinance: API for accessing Yahoo Finance data. 
        pandas, numpy: Data manipulation and analysis. 
        plotly, matplotlib: Data visualization. 
        TextBlob: Sentiment analysis library.

### How to install the required packages
        pip install openai yfinance pandas matplotlib streamlit

### How to run the program
        Go to the program location -> Open cmd -> python -m streamlit run main.py

### Explanation of Functions and Code
#####   Data Retrieval Functions:
        get_stock_price(ticker): Fetches the current stock price.
        calculate_sma(data, window): Calculates the Simple Moving Average.
        calculate_ema(data, window): Calculates the Exponential Moving Average.
        calculate_rsi(data, window): Calculates the Relative Strength Index.
        analyze_sentiment(ticker): Performs sentiment analysis.
        Investment Recommendation Functions:

#####  recommend_investment(rsi, sentiment_score): Recommends Buy, Sell, or Hold based on RSI and sentiment.
        suggest_investment_amount(current_price, risk_percentage): Suggests the investment amount based on risk tolerance.
        calculate_win_loss_probability(sentiment_score): Calculates win and loss probabilities.
        Performance Metrics Functions:

##### calculate_sharpe_ratio(data, risk_free_rate): Calculates the Sharpe ratio.
        calculate_sortino_ratio(data, risk_free_rate): Calculates the Sortino ratio.
        calculate_max_drawdown(data): Calculates the Maximum Drawdown.

##### Streamlit App Setup: Page configuration, background color, and styling.
        User interface elements for stock selection, sliders, and financial condition input.
        Display of stock information, technical indicators, sentiment analysis, and investment recommendations.
        Visualization of historical data using line charts.
        Output of personalized investment suggestions based on user input.

