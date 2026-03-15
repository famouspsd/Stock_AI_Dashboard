
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="Stock AI", layout="wide")
st.title("🚀 Professional Stock AI Dashboard")

ticker = st.sidebar.text_input("Enter Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["2y", "5y"])

@st.cache_data
def load_data(symbol, p):
    data = yf.download(symbol, period=p)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df = load_data(ticker, period)

# Features
df['SMA_20'] = df['Close'].rolling(20).mean()
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
df['Target'] = df['Close'].shift(-1)
df_model = df.dropna()

# AI Model
model = RandomForestRegressor(n_estimators=100)
model.fit(df_model[['SMA_20', 'RSI']], df_model['Target'])
prediction = model.predict(df[['SMA_20', 'RSI']].tail(1))[0]

# --- NEW: Recommendation Logic ---
current_price = df['Close'].iloc[-1]
rsi_val = df['RSI'].iloc[-1]

if rsi_val < 35:
    recommendation = "STRONG BUY (Oversold)"
    color = "green"
elif rsi_val > 65:
    recommendation = "STRONG SELL (Overbought)"
    color = "red"
else:
    recommendation = "HOLD / NEUTRAL"
    color = "grey"

# Layout
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("AI Prediction", f"${prediction:.2f}", f"{prediction-current_price:.2f}")
col3.markdown(f"### Signal: :{color}[{recommendation}]")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name="Trend (SMA)"))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

st.success(f"The AI model suggests a price movement of {((prediction/current_price)-1)*100:.2f}% for the next trading day.")
    