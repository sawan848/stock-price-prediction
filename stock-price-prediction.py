#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
import pandas_datareader.data as web
from keras.layers import Dense,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler


# In[2]:


start_date=datetime(2000, 1, 1)


# In[3]:


stock='AAPL'
AAPL_df=web.DataReader(stock,data_source='yahoo',start=start_date)


# In[4]:


AAPL_df


# In[5]:


#1 reset the date indec to column
AAPL_df=AAPL_df.reset_index()


# In[6]:


AAPL_df


# In[7]:


AAPL_df.describe()


# In[8]:


AAPL_df


# In[9]:


# 2 .Drop null values
AAPL_df.dropna(inplace = True)


# In[10]:


AAPL_df.describe()


# In[11]:


#3.sort the datavalues by date
AAPL_df.sort_values(by = 'Date', ignore_index = True)


# In[12]:


AAPL_df


# In[13]:


#4 Drop rows having Date < '2015-01-01'
AAPL_df=AAPL_df[AAPL_df['Date']>='2015-01-01'].reset_index(drop=True)


# In[14]:


AAPL_df.describe()


# In[15]:


#5 Change Dtype of Columns
AAPL_df['Date']=pd.to_datetime(AAPL_df["Date"])
    
for col in ["Open","High","Low","Close","Adj Close","Volume"]:
    AAPL_df[col]=AAPL_df[col].astype(str).str.replace(',','')


# In[16]:


AAPL_df=AAPL_df.astype({"Open":float,"High":float,"Low":float,"Close":float,"Adj Close":float,"Volume":float})


# In[17]:


AAPL_df


# In[18]:


stock='TSLA'
TSLA_df=web.DataReader(stock,data_source='yahoo',start=start_date)


# In[19]:


stock='AMZN'
AMZN_df=web.DataReader(stock,data_source='yahoo',start=start_date)


# In[20]:


stock='GOOG'
GOOG_df=web.DataReader(stock,data_source='yahoo',start=start_date)


# In[21]:


stock='MSFT'
MSFT_df=web.DataReader(stock,data_source='yahoo',start=start_date)


# In[22]:


#1 reset the date indec to column
MSFT_df=AAPL_df.reset_index()
TSLA_df=TSLA_df.reset_index()
GOOG_df=GOOG_df.reset_index()
AMZN_df=MSFT_df.reset_index()


# In[23]:


# 2 .Drop null values
MSFT_df.dropna(inplace=True)
TSLA_df.dropna(inplace=True)
GOOG_df.dropna(inplace=True)
AMZN_df.dropna(inplace=True)


# In[24]:


#3.sort the datavalues by date
MSFT_df.sort_values(by = 'Date', ignore_index = True)
TSLA_df.sort_values(by = 'Date', ignore_index = True)
GOOG_df.sort_values(by = 'Date', ignore_index = True)
AMZN_df.sort_values(by = 'Date', ignore_index = True)


# In[25]:


#4 Drop rows having Date < '2015-01-01'
MSFT_df=AAPL_df[AAPL_df['Date']>='2015-01-01'].reset_index(drop=True)
TSLA_df=AAPL_df[AAPL_df['Date']>='2015-01-01'].reset_index(drop=True)
GOOG_df=AAPL_df[AAPL_df['Date']>='2015-01-01'].reset_index(drop=True)
AMZN_df=AAPL_df[AAPL_df['Date']>='2015-01-01'].reset_index(drop=True)


# In[26]:


#5 Change Dtype of Columns MSFT_df
MSFT_df['Date']=pd.to_datetime(MSFT_df["Date"])
    
for col in ["Open","High","Low","Close","Adj Close","Volume"]:
    MSFT_df[col]=MSFT_df[col].astype(str).str.replace(',','')

MSFT_df=MSFT_df.astype({"Open":float,"High":float,"Low":float,"Close":float,"Adj Close":float,"Volume":float})


# In[27]:


# Change Dtype of Columns TSLA_df
TSLA_df['Date']=pd.to_datetime(TSLA_df["Date"])
    
for col in ["Open","High","Low","Close","Adj Close","Volume"]:
    TSLA_df[col]=TSLA_df[col].astype(str).str.replace(',','')

TSLA_df=TSLA_df.astype({"Open":float,"High":float,"Low":float,"Close":float,"Adj Close":float,"Volume":float})


# In[28]:


# Change Dtype of Columns AMZN_df
AMZN_df['Date']=pd.to_datetime(AMZN_df["Date"])
    
for col in ["Open","High","Low","Close","Adj Close","Volume"]:
    AMZN_df[col]=AMZN_df[col].astype(str).str.replace(',','')

AMZN_df=AMZN_df.astype({"Open":float,"High":float,"Low":float,"Close":float,"Adj Close":float,"Volume":float})


# In[29]:


# Change Dtype of Columns GOOG_df
GOOG_df['Date']=pd.to_datetime(GOOG_df["Date"])
    
for col in ["Open","High","Low","Close","Adj Close","Volume"]:
    GOOG_df[col]=GOOG_df[col].astype(str).str.replace(',','')

GOOG_df=GOOG_df.astype({"Open":float,"High":float,"Low":float,"Close":float,"Adj Close":float,"Volume":float})


# # Exploratory Data Analysis

# In[30]:


Companies=[AAPL_df,MSFT_df,TSLA_df,AMZN_df,GOOG_df]
Companies_Title=["Apple","Microsoft","Tesla","Amazon","Google"]


# In[31]:


#6 Now lets plot the total volume of stock being traded each day
plt.figure(figsize=(20,20))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company["Date"],company["Volume"])
    plt.title(Companies_Title[index])
    plt.ylabel('Volume')


# ### What was the moving average of the various stocks ?

# In[32]:


#7 calculate Moving Average
Moving_Average_Day=[10,20,50]

for Moving_Average in Moving_Average_Day:
    for company in Companies:
        column_name=f'Moving Average for {Moving_Average} days'
        company[column_name]=company['Adj Close'].rolling(Moving_Average).mean()


# In[33]:


plt.figure(figsize=(20,12))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company["Date"], company["Adj Close"])
    plt.plot(company["Date"], company["Moving Average for 10 days"])
    plt.plot(company["Date"], company["Moving Average for 20 days"])
    plt.plot(company["Date"], company["Moving Average for 50 days"])
    plt.title(Companies_Title[index])
    plt.legend(("Adj Close","Moving Averge for 10 days","Moving Averge for 20 days","Moving Averge for 50 days"))


# In[34]:


Moving_Average_Day2=[100,200,500]

for Moving_Average in Moving_Average_Day2:
    for company in Companies:
        column_name=f'Moving Average for {Moving_Average} days'
        company[column_name]=company['Adj Close'].rolling(Moving_Average).mean()


# In[35]:


plt.figure(figsize=(20,12))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company["Date"], company["Adj Close"])
    plt.plot(company["Date"], company["Moving Average for 100 days"])
    plt.plot(company["Date"], company["Moving Average for 200 days"])
    plt.plot(company["Date"], company["Moving Average for 500 days"])
    plt.title(Companies_Title[index])
    plt.legend(("Adj Close","Moving Averge for 10 days","Moving Averge for 20 days","Moving Averge for 50 days"))


# In[36]:


ma10=AAPL_df.Close.rolling(10).mean()


# In[37]:


ma10


# In[38]:


ma20=AAPL_df.Close.rolling(20).mean()
ma20


# In[39]:


ma50=AAPL_df.Close.rolling(50).mean()
ma50


# In[ ]:





# In[40]:


plt.figure(figsize=(20,20))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company["Date"], company["Adj Close"])
    plt.plot(company["Date"], ma10)
    plt.plot(company["Date"], ma20)
    plt.plot(company["Date"], ma50)
    plt.title(Companies_Title[index])
    plt.legend(("Adj Close","Moving Averge for 10 days","Moving Averge for 20 days","Moving Averge for 50 days"))


# In[41]:


plt.figure(figsize=(10,8))
plt.plot(AAPL_df.Close)
plt.plot(ma10,'r')
plt.plot(ma20,'g')
plt.plot(ma50,'y')


# In[42]:


ma100=AAPL_df.Close.rolling(100).mean()
ma200=AAPL_df.Close.rolling(200).mean()
ma500=AAPL_df.Close.rolling(500).mean()


# In[43]:


plt.figure(figsize=(10,8))
plt.plot(AAPL_df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(ma500,'y')


# In[44]:


AAPL_df


# ### What was the daily return of the stock on average ?

# In[45]:


# 8. daily return
for company in Companies:
    company["Daily Return"]=company["Adj Close"].pct_change()


# In[46]:


plt.figure(figsize=(20,20))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index + 1)
    plt.plot(company["Date"],company["Daily Return"])
    plt.title(Companies_Title[index])
    plt.ylabel("Daily Return")


# In[47]:


# distplot is a deprecated function, so to ignore warnings, the filterwarnings function is used.
# 9. get an overall at the average daily return using a histogram
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(20, 20))
for index, company in enumerate(Companies):
  plt.subplot(3, 2, index + 1)
  sns.distplot(company["Daily Return"].dropna(), color = "purple")
  plt.title(Companies_Title[index])


# In[48]:


#10 find kurtosis value
print("Kurtosis value")
for index,company in enumerate(Companies):
    print(f'{Companies_Title[index]}: {company["Daily Return"].kurtosis()}')


# In[49]:


# 11 find out correlation between stocks closing price

Companies_returns=pd.DataFrame()
Companies_returns["AAPL"]=AAPL_df["Adj Close"]
Companies_returns["TSLA"]=TSLA_df["Adj Close"]
Companies_returns["GOOG"]=GOOG_df["Adj Close"]
Companies_returns["MSFT"]=MSFT_df["Adj Close"]
Companies_returns["AMZN"]=AMZN_df["Adj Close"]
Companies_returns.head()


# In[50]:


#12 Companies_Daily_returns contains percentage daily returns of all the companies
Companies_Daily_returns = Companies_returns.pct_change()
Companies_Daily_returns.head()


# In[51]:


sns.heatmap(Companies_returns.corr(),annot=True,cmap="YlGnBu")


# In[52]:


Companies_returns.corr()


# In[53]:


sns.heatmap(Companies_Daily_returns.corr(),annot=True,cmap="YlGnBu")


# In[54]:


Companies_Daily_returns.corr()


# In[55]:


Return=Companies_Daily_returns.dropna()

plt.figure(figsize=(10,5))
plt.scatter(Return.mean(),Return.std())
plt.xlabel('Expected Return')
plt.ylabel("Risk")

for label,x,y in zip(Companies_Title,Return.mean(),Return.std()):
    plt.annotate(label,xy=(x,y),xytext=(10,0),textcoords='offset points')


# In[56]:


# !pip install pmdarima


# In[57]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[58]:


# !pip install pyramid


# In[59]:


# !pip install pmdarima


# In[60]:


#!pip install pmdarima


# In[61]:


#!pip install pyramid-arima


# In[62]:


# !pip install pmdarima


# In[63]:


import  scipy.signal.signaltools


# In[64]:


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered


# In[65]:


from pmdarima.arima import auto_arima


# In[66]:


#Arima model


# In[67]:


def Test_Stationarity(timeseries):
    resullt=adfuller(timeseries['Adj Close'],autolag="AIC")
    print("Results of Dickey Fuller Test")
    print(f'Test Statistics: {resullt[0]}')
    print(f'p-value: {resullt[1]}')
    print(f'Number of lags used: {resullt[2]}')
    print(f'Number of observations used: {resullt[3]}')
    
    for key,value in resullt[4].items():
        print(f'critical value ({key}): {value}')


# In[68]:


AAPL_df.info()


# In[69]:


AAPL_df["Date"]=pd.to_datetime(AAPL_df["Date"])


# In[70]:


Test_Stationarity(AAPL_df)


# Now let's take log of the 'Adj. Close' column to reduce the magnitude of the values and reduce the series rising trend.

# In[71]:


AAPL_df['log Adj Close']=np.log(AAPL_df['Adj Close'])
AAPL_log_moving_avg=AAPL_df['log Adj Close'].rolling(12).mean()
AAPL_log_std=AAPL_df['log Adj Close'].rolling(12).std()

plt.figure(figsize=(10,5))
plt.plot(AAPL_df['Date'],AAPL_log_moving_avg,label='Rolling Mean')
plt.plot(AAPL_df['Date'],AAPL_log_std,label='Rolling  Std')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')


# ### Split the data into training and test set
# 
#     Training Period: 2015-01-02 - 2020-09-30
# 
#     Testing Period:  2020-10-01 - 2021-02-26

# In[72]:


AAPL_Train_Data=AAPL_df[AAPL_df['Date'] < '2020-10-01']
AAPL_Test_Data=AAPL_df[AAPL_df['Date']>= '2020-10-01'].reset_index(drop=True)


# In[73]:


plt.figure(figsize=(10,5))
plt.plot(AAPL_Train_Data['Date'],AAPL_Train_Data['log Adj Close'],label='Train Data')
plt.plot(AAPL_Test_Data['Date'],AAPL_Test_Data['log Adj Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')


# In[74]:


AAPL_Train_Data


# In[75]:


AAPL_Test_Data


# In[76]:


data_training=pd.DataFrame(AAPL_df['Close'][0:int(len(AAPL_df)*0.70)])
data_testing=pd.DataFrame(AAPL_df['Close'][int(len(AAPL_df)*0.70):int(len(AAPL_df))])


# In[77]:


data_testing


# In[78]:


data_training


# In[79]:


plt.figure(figsize=(10,5))
plt.plot(data_training['Close'],data_training['Close'],label='Train Data')
plt.plot(data_testing['Close'],data_testing['Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('Close')
plt.legend(loc='best')


# ### Arima Model

# In[80]:


AAPL_Auto_Arima_Model_Adj_Close=auto_arima(AAPL_Train_Data['log Adj Close'],seasonal=False
                                ,error_action='ignore' ,suppress_warnings=True)
print(AAPL_Auto_Arima_Model_Adj_Close.summary())


# In[81]:


AAPL_Auto_Arima_Model_Close=auto_arima(data_training['Close'],seasonal=False
                                ,error_action='ignore' ,suppress_warnings=True)
print(AAPL_Auto_Arima_Model_Close.summary())


# In[82]:


AAPL_Arima_Model_Adj_Close=ARIMA(AAPL_Train_Data['log Adj Close'],order=(1,1,0))
AAPL_Arima_Model_Adj_Close_Fit=AAPL_Arima_Model_Adj_Close.fit()
print(AAPL_Arima_Model_Adj_Close_Fit.summary())


# In[83]:


AAPL_Arima_Model_Close=ARIMA(data_training['Close'],order=(1,1,0))
AAPL_Arima_Model_Close_Fit=AAPL_Arima_Model_Close.fit()
print(AAPL_Arima_Model_Close_Fit.summary())


# ### Predicting the closing stock price of Apple

# In[84]:


# AAPL_output=AAPL_Arima_Model_Adj_Close_Fit.forecast(102,alpha=0.05)
# AAPL_predictions=np.exp(AAPL_output[0])

# plt.figure(figsize=(10,5))

# plt.plot(AAPL_Train_Data['Date'],AAPL_Train_Data['Adj Close'],label='Training')
# plt.plot(AAPL_Test_Data['Date'],AAPL_Test_Data['Adj Close'],label='Testing')
# plt.plot(AAPL_Test_Data['Date'],AAPL_predictions, label='Predictions')
# plt.plot('Time')
# plt.plot('Closing Price')
# plt.legend()


# In[85]:


AAPL_output = AAPL_Arima_Model_Adj_Close_Fit.forecast(102, alpha=0.05)
AAPL_predictions = np.exp(AAPL_output[0])


# In[86]:


AAPL_predictions


# In[87]:


AAPL_output


# In[88]:


# plt.figure(figsize=(10, 5))
# plt.plot(AAPL_Train_Data['Date'], AAPL_Train_Data['Adj Close'], label = 'Training')
# plt.plot(AAPL_Test_Data['Date'], AAPL_Test_Data['Adj Close'], label = 'Testing')
# plt.plot(AAPL_Test_Data['Date'], AAPL_predictions, label = 'Predictions')
# plt.xlabel('Time')
# plt.ylabel('Closing Price')
# plt.legend()


# In[89]:


print(AAPL_Test_Data.shape)
print(AAPL_predictions.shape)


# In[90]:


print(AAPL_Test_Data['Date'])


# In[91]:


print(AAPL_predictions)


# In[92]:


# import math
# rmse = math.sqrt(mean_squared_error(AAPL_Test_Data['Adj Close'], AAPL_predictions))
# mape = np.mean(np.abs(AAPL_predictions - AAPL_Test_Data['Adj Close'])/np.abs(AAPL_Test_Data['Adj Close']))

# print(f'RMSE: {rmse}')
# print(f'MAPE: {mape}')


# In[93]:


# import math
# rmse = math.sqrt(mean_squared_error(AAPL_Test_Data['Close'], AAPL_predictions))
# # mape = np.mean(np.abs(AAPL_predictions - AAPL_Test_Data['Adj Close'])/np.abs(AAPL_Test_Data['Adj Close']))

# print(f'RMSE: {rmse}')
# # print(f'MAPE: {mape}')


# In[94]:


AAPL_Test_Data['Adj Close']


# In[ ]:




