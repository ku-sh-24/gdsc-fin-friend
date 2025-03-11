# # import yfinance as yf
# # import pandas as pd
# # #import matplotlib.pyplot as plt

# # ticker = "AAPL"
# # stock = yf.Ticker(ticker)

# # stock_info = stock.info
# # print(stock_info)

# # AIzaSyAtOqaJNh-ebtGVTRvBHxZgxHUDbIKgzdI

# import yfinance as yf
# from google import genai

# def analyze_stock_with_llm(ticker, api_key):
#     """
#     Fetches stock data from yfinance, converts it to a string,
#     and sends it to a LLM for analysis.

#     Args:
#         ticker: The stock ticker symbol (e.g., "AAPL").
#         project_id: Your Google Cloud project ID.
#         location: The location of your Vertex AI endpoint.
#         endpoint_id: The ID of your deployed Vertex AI endpoint.
#     """
#     try:
#         stock = yf.Ticker(ticker)

#         # Extract relevant data (adjust as needed)
#         info = stock.info
#         history = stock.history(period="1y") # Or another period
#         financials = stock.financials
#         balance_sheet = stock.balance_sheet
#         cashflow = stock.cashflow
#         sustainability = stock.sustainability
#         recommendations = stock.recommendations

#         # Create a string representation of the data
#         data_string = f"Stock Analysis for {ticker}:\n\n"
#         data_string += "Company Info:\n" + str(info) + "\n\n"
#         data_string += "Historical Data:\n" + str(history) + "\n\n"
#         data_string += "Financials:\n" + str(financials) + "\n\n"
#         data_string += "Balance Sheet:\n" + str(balance_sheet) + "\n\n"
#         data_string += "Cashflow:\n" + str(cashflow) + "\n\n"
#         data_string += "Sustainability:\n" + str(sustainability) + "\n\n"
#         data_string += "Recommendations:\n" + str(recommendations) + "\n\n"

#         client = genai.Client(api_key=api_key)

#         response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=[f"Analyze the following stock data and provide an overall analysis in brief: {data_string}"])

#         # genai.configure(api_key=api_key)
#         # model = genai.GenerativeModel('gemini-pro') #Or gemini-ultra if you have access.
#         # prompt = f"Analyze the following stock data and provide an overall analysis: {data_string}"

#         # Generate the response
#         # response = model.generate_content(prompt)
#         print(response.text)


#         # Send to LLM (using Vertex AI as an example)
#         # client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
#         # predict_client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

#         # endpoint = predict_client.endpoint_path(
#         #     project=project_id, location=location, endpoint=endpoint_id
#         # )

#         # instance = {"prompt": f"Analyze the following stock data and provide an overall analysis: {data_string}"}
#         # instances = [instance]

#         # response = predict_client.predict(endpoint=endpoint, instances=instances)

#         # for prediction in response.predictions:
#         #     print(prediction["content"])

#     except Exception as e:
#         print(f"Error: {e}")

# # Example usage (replace with your values)
# ticker_symbol = "AAPL"
# api_key = "AIzaSyAtOqaJNh-ebtGVTRvBHxZgxHUDbIKgzdI"

# analyze_stock_with_llm(ticker_symbol, api_key)

# #If using OpenAI, replace the vertex AI section with:
# # import openai
# # openai.api_key = "YOUR_OPENAI_API_KEY"
# # response = openai.Completion.create(
# #   engine="text-davinci-003", #Or another engine
# #   prompt = f"Analyze the following stock data and provide an overall analysis: {data_string}",
# #   max_tokens = 1024,
# # )
# # print(response.choices[0].text.strip())

import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]  # Prophet requires columns named ds (date) and y (value)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df['y'] = df['y'].astype(float)
    return df

def train_prophet(df, future_days):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    return fig

# Streamlit App
st.title("📈 Stock Forecasting Assistant")
st.write("Hello! I'm your stock forecasting assistant. Ask me about a stock, and I'll predict its future prices.")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", "AAPL")
future_days = st.slider("Select forecast duration (days):", 7, 365, 30)

if st.button("Predict Stock Prices"):
    st.write(f"Fetching data for {ticker} and predicting {future_days} days ahead...")
    df = fetch_stock_data(ticker)
    model, forecast = train_prophet(df, future_days)
    
    st.write("Here's the forecast:")
    st.plotly_chart(plot_forecast(df, forecast))
    
    st.write("### Key Insights:")
    st.write(f"- *Trend*: The stock shows {'an upward' if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] else 'a downward'} trend over the forecast period.")
    st.write("- *Uncertainty*: The confidence intervals indicate potential price fluctuations.")
    st.write("- *Seasonality Effects*: Explore seasonality patterns below.")
    
    st.write("### Seasonality Trends")
    fig_seasonality = model.plot_components(forecast)
    st.pyplot(fig_seasonality)