from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st


#df = pd.read_csv("AAPL.csv")
start='2010-01-01'
end='2021-12-31'

st.title('Stock Market Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)
#df=data.DataReader('AAPL','yahoo',start,end)
#df=yf.download('AAPL',start,end)

# describing data

st.subheader('Data From 2010 - 2021')
st.write(df.describe())

# visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma200, 'r')
plt.plot(ma100, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting Data into Training and Testing

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_train.shape)
print(data_test.shape)

scaler = MinMaxScaler(feature_range=(0, 1))

data_train_arr = scaler.fit_transform(data_train)

# Load my model

model = load_model('stock_prediction.h5')

# Testing Part

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predict = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor

# Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Orignal Price')
plt.plot(y_predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
