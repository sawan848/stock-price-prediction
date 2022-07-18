from unicodedata import name
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import cufflinks as cf
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt 
from keras.models import load_model
import pandas_datareader.data as web
from plotly import graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
import datetime

# App title

st.set_page_config(page_title="Stock Prediction App")

st.markdown("""<style>#root>div:nth-child(1)>div>div>div>div>section>div {color:red}
        </style>
        """,unsafe_allow_html=True)

st.markdown('''
# Stock Price App
Shown are the stock price data for query companies!
**Credits**
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
''')
st.write('---')

@st.cache
def load_data(ticker):
    data=web.DataReader(ticker,data_source='yahoo',start=start_date)
    data.reset_index(inplace=True)
    return data


# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2005, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2022, 4, 30))
n_years=st.slider("years of predictions: ",1,4)
period=n_years*365


# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerSymbol=st.sidebar.text_input('Enter Stock Ticker','SBIN.NS')
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker

#Apple --> AAPL
#Statebank--> SBIN.NS

# st.write('---')
# st.write(tickerData.info)


# # Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# # Ticker data


data=load_data(tickerSymbol)
st.header('**Ticker data**')
st.write(data)
# st.write(tickerDf)




# #describing data
st.subheader('Data from 2005 - Till Now')
st.write(data.describe())

# # Time series Data
def plot_row_data():
    fig=go.Figure(layout=go.Layout(height=600,width=900))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_row_data()

st.subheader('Closing Price vs time chart 100MA ')
ma100=data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs time chart 100MA & 200MA')
ma100=data.Close.rolling(100).mean()
ma200=data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close,'b')
plt.legend()
st.pyplot(fig)

# # Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands(periods=20,boll_std=2,colors=['cyan','gray'],fill=True)
# qf.add_volume(name='volume',up_color='green',down_color='red')
fig = qf.iplot(asFigure=True)

fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
st.plotly_chart(fig)



# #forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)

future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail()) 


st.header("Forecast data")
fig1=plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.header("Forecast components")
fig2=plot_components_plotly(m,forecast)
st.plotly_chart(fig2)




data_training=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


# #load model
model =load_model('keras_model.h5')

# #testing part

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
 
x_test,y_test=np.array(x_test),np.array(y_test)


# # making predictions
y_predicted=model.predict(x_test)
scaler=scaler.scale_
scaler_factor=1/scaler[0]
y_predicted=y_predicted*scaler_factor
y_test=y_test*scaler_factor


# #final graph

st.subheader('Predictions vs Original ')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



####
# 