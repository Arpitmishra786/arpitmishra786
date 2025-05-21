import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ----- Streamlit Setup -----
st.set_page_config(page_title="Crypto Trading Platform", layout="wide")
st_autorefresh(interval=15000, key="refresh")
st.title("🚀 Crypto Trading Platform")

# ----- Functions -----
def get_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return float(data['price'])
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return None

def get_ohlcv(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data: {e}")
        return pd.DataFrame()

def add_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    return df

def add_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def generate_signals(df):
    df['signal'] = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'Buy',
                     np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'Sell', ''))
    return df

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# ----- UI -----
symbol = st.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"])
interval = st.selectbox("Select Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])

price = get_price(symbol)
if price is not None:
    st.metric(label=f"💰 {symbol} Live Price", value=f"${price:,.2f}")
else:
    st.warning("Price data not available.")

# ----- Load Data -----
ohlcv_df = get_ohlcv(symbol, interval=interval, limit=200)
if ohlcv_df.empty:
    st.warning("OHLCV data not available. Please try again later.")
    st.stop()

ohlcv_df = add_macd(ohlcv_df)
ohlcv_df = add_rsi(ohlcv_df)
ohlcv_df = generate_signals(ohlcv_df)

# ----- Chart -----
st.subheader("📊 Live Candlestick Chart with MACD Signals")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=ohlcv_df['timestamp'], open=ohlcv_df['open'], high=ohlcv_df['high'],
    low=ohlcv_df['low'], close=ohlcv_df['close'], name='Candles'))

buy_signals = ohlcv_df[ohlcv_df['signal'] == 'Buy']
sell_signals = ohlcv_df[ohlcv_df['signal'] == 'Sell']

fig.add_trace(go.Scatter(x=buy_signals['timestamp'], y=buy_signals['close'],
                         mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
fig.add_trace(go.Scatter(x=sell_signals['timestamp'], y=sell_signals['close'],
                         mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))

fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# ----- LSTM Prediction -----
close_prices = ohlcv_df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_prices)

seq_len = 10
X, y = create_sequences(scaled_close, seq_len)
if len(X) > 0:
    X = X.reshape((X.shape[0], seq_len, 1))
    model = Sequential()
    model.add(Input(shape=(seq_len, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("Training LSTM model..."):
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)

    # Predict future
    last_seq = scaled_close[-seq_len:].reshape((1, seq_len, 1))
    pred_scaled = model.predict(last_seq, verbose=0)
    predicted_price = scaler.inverse_transform(pred_scaled)[0][0]
    st.success(f"📈 Predicted Next Close Price: ${predicted_price:,.2f}")

    # Plot historical vs predicted
    st.subheader("🔮 Historical Predictions (last 50 points)")
    y_pred = model.predict(X, verbose=0)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=y_true[-50:].flatten(), name='Actual'))
    fig2.add_trace(go.Scatter(y=y_pred_inv[-50:].flatten(), name='Predicted'))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("❗Not enough data to train LSTM model. Try increasing limit or wait for more candles.")


