#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler


# In[2]:


AAPL=pd.read_csv("Database/AAPL.csv")
TSLA=pd.read_csv("Database/TSLA.csv")
GOOG=pd.read_csv("Database/GOOG.csv")
MSFT=pd.read_csv("Database/MSFT.csv")
AMZN=pd.read_csv("Database/AMZN.csv")


# In[3]:


AAPL.head()


# In[4]:


AAPL.info()


# In[5]:


AAPL.dropna(inplace=True)


# In[6]:


#change Dtype of column
AAPL["Date"]=pd.to_datetime(AAPL["Date"])
AAPL["Volume"]=AAPL["Volume"].str.replace(',','')
AAPL=AAPL.astype({"Open":float,"Volume":float})


# In[7]:


#sort the dataset by date
AAPL=AAPL.sort_values(by='Date',ignore_index=True)


# In[8]:


AAPL


# In[9]:


#Drop rows having Date < '2015-01-01'
AAPL=AAPL[AAPL['Date']>='2015-01-01'].reset_index(drop=True)


# In[10]:


AAPL.describe()


# In[11]:


TSLA.dropna(inplace=True)
GOOG.dropna(inplace=True)
AMZN.dropna(inplace=True)
MSFT.dropna(inplace=True)


# ### change Dtype of column

# In[12]:


TSLA["Date"]=pd.to_datetime(TSLA["Date"])
TSLA["Volume"]=TSLA["Volume"].str.replace(',','')
TSLA=TSLA.astype({"Open":float,"Volume":float})


# In[13]:


GOOG["Date"]=pd.to_datetime(GOOG["Date"])
for col in ["Open","High","Low","Close","Adj. Close","Volume"]:
    GOOG[col]=GOOG[col].str.replace(',','')
    
GOOG=GOOG.astype({"Open":float,"High":float,
                  "Low":float,'Close':float,
                  'Adj. Close':float,"Volume":float})


# In[14]:


MSFT["Date"]=pd.to_datetime(MSFT["Date"])
MSFT['Open']=MSFT['Open'].str.replace(',','')
MSFT["Volume"]=MSFT["Volume"].str.replace(',','')
MSFT=MSFT.astype({"Open":float,"Volume":float})


# In[15]:


AMZN["Date"]=pd.to_datetime(MSFT["Date"])

for col in ['Open','High','Low','Close','Adj. Close','Volume']:
    AMZN[col]=AMZN[col].str.replace(',','')

AMZN=AMZN.astype({"Open":float,"High":float,
                  "Low":float,"Close":float,
                  "Adj. Close":float,"Volume":float})


# ### sort the dataset by date

# In[16]:


AMZN=AMZN.sort_values(by='Date',ignore_index=True)
MSFT=MSFT.sort_values(by='Date',ignore_index=True)
GOOG=GOOG.sort_values(by='Date',ignore_index=True)
TSLA=TSLA.sort_values(by='Date',ignore_index=True)


# ### Drop rows having Date < '2015-01-01'

# In[17]:


TSLA=TSLA[TSLA['Date']>='2015-01-01'].reset_index(drop=True)
GOOG=GOOG[GOOG['Date']>='2015-01-01'].reset_index(drop=True)
MSFT=MSFT[MSFT['Date']>='2015-01-01'].reset_index(drop=True)
AMZN=AMZN[AMZN['Date']>='2015-01-01'].reset_index(drop=True)


# # Exploratory Data Analysis

# In[18]:


Companies=[AAPL,TSLA,GOOG,MSFT,AMZN]
Companies_Title=['Apple','Tesla','Google','Microsoft','Amazon']


# In[19]:


plt.figure(figsize=(20,20))

for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company['Date'],company['Volume'])
    plt.title(Companies_Title[index])
    plt.ylabel('Volume')


# ### What was the moving average of the various stocks ?

# In[20]:


Moving_Average_Days=[10,20,50]

for Moving_Average in Moving_Average_Days:
    for company in Companies:
        column_name=f'Moving Average for {Moving_Average} days'
        company[column_name]=company['Adj. Close'].
        rolling(Moving_Average).mean()


# In[21]:


plt.figure(figsize=(20,20))

for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company['Date'],company['Adj. Close'])
    plt.plot(company['Date'],company['Moving Average for 10 days'])
    plt.plot(company['Date'],company['Moving Average for 20 days'])
    plt.plot(company['Date'],company['Moving Average for 50 days'])
    plt.title(Companies_Title[index])
    plt.legend(('Adj. Close','Moving Average for 10 days',
                'Moving Average for 20 days','Moving Average for 50 days'))


# In[22]:


Moving_Average_Days_2=[100,200,500]

for Moving_Average in Moving_Average_Days_2:
    for company in Companies:
        column_name=f'Moving Average for {Moving_Average} days'
        company[column_name]=company['Adj. Close'].
        rolling(Moving_Average).mean()


# In[23]:


plt.figure(figsize=(20,20))

for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company['Date'],company['Adj. Close'])
    plt.plot(company['Date'],company['Moving Average for 100 days'])
    plt.plot(company['Date'],company['Moving Average for 200 days'])
    plt.plot(company['Date'],company['Moving Average for 500 days'])
    plt.title(Companies_Title[index])
    plt.legend(('Adj. Close','Moving Average for 100 days',
                'Moving Average for 200 days',
                'Moving Average for 500 days'))


# ### What was the daily return of the stock on average ?

# In[24]:


for company in Companies:
    company['Daily Return']=company['Adj. Close'].pct_change()


# In[25]:


plt.figure(figsize=(20,20))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    plt.plot(company['Date'],company['Daily Return'])
    plt.title(Companies_Title[index])
    plt.ylabel('Daily Return')


# In[26]:


import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(20,20))
for index,company in enumerate(Companies):
    plt.subplot(3,2,index+1)
    sns.distplot(company['Daily Return'].dropna(),color='purple')
    plt.title(Companies_Title[index])


# In[27]:


print("Kurtosis value")
for index,company in enumerate(Companies):
    print(f'{Companies_Title[index]}: {company["Daily Return"].kurtosis()}')


# ### What was the correlation between diffrent stocks closing price ?

# In[28]:


Companies_returns=pd.DataFrame()
Companies_returns['AAPL']=AAPL['Adj. Close']
Companies_returns['TSLA']=TSLA['Adj. Close']
Companies_returns['MSFT']=MSFT['Adj. Close']
Companies_returns['GOOG']=GOOG['Adj. Close']
Companies_returns['AMZN']=AMZN['Adj. Close']


# In[29]:


Companies_returns.head()


# In[30]:


Companies_Daily_returns=Companies_returns.pct_change()
Companies_Daily_returns.head()


# In[31]:


sns.heatmap(Companies_returns.corr(),annot=True,cmap='YlGnBu')


# In[32]:


Companies_returns.corr()


# In[33]:


Companies_Daily_returns.corr()


# In[34]:


sns.heatmap(Companies_Daily_returns.corr(),annot=True,cmap='YlGnBu')


# In[35]:


Return=Companies_Daily_returns.dropna()

plt.figure(figsize=(20,5))
plt.scatter(Return.mean(),Return.std())
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label,x,y in zip(Companies_Title,Return.mean(),Return.std()):
    plt.annotate(label,xy=(x,y),xytext=(10,0),textcoords='offset points')


# In[36]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error
import  scipy.signal.signaltools
import warnings
warnings.filterwarnings('ignore')


# In[37]:


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered


# In[38]:


from pmdarima.arima import auto_arima


# In[39]:


Pre_Processed_AAPL=pd.read_csv("Database/Pre_Processed_AAPL.csv")
Pre_Processed_TSLA=pd.read_csv("Database/Pre_Processed_TSLA.csv")
Pre_Processed_GOOG=pd.read_csv("Database/Pre_Processed_GOOG.csv")
Pre_Processed_MSFT=pd.read_csv("Database/Pre_Processed_MSFT.csv")
Pre_Processed_AMZN=pd.read_csv("Database/Pre_Processed_AMZN.csv")


# In[40]:


Pre_Processed_AAPL


# In[41]:


AAPL


# In[42]:


def Test_Stationarity(timeseries):
    result=adfuller(timeseries['Adj. Close'],autolag='AIC')
    print('Results of Dickey Fuller Test')
    print(f'Test Statistics: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Number of legs used: {result[2]}')
    print(f'Number of Observations used: {result[3]}')
    for key,value in result[4].items():
        print(f'critical value ({key}):{value}')
    


# In[43]:


Pre_Processed_AAPL.info()


# In[44]:


Pre_Processed_AAPL.head()


# In[45]:


Pre_Processed_AAPL['Date']=pd.to_datetime(Pre_Processed_AAPL["Date"])


# In[46]:


Pre_Processed_AAPL.head()


# In[47]:


Pre_Processed_AAPL


# In[48]:


Test_Stationarity(Pre_Processed_AAPL)


# In[49]:


Pre_Processed_AAPL['log Adj Close']=np.log(Pre_Processed_AAPL['Adj. Close'])
AAPL_log_moving_avg=Pre_Processed_AAPL['log Adj Close'].rolling(12).mean()
AAPL_log_std=Pre_Processed_AAPL['log Adj Close'].rolling(12).std()

plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_AAPL['Date'],AAPL_log_moving_avg,label='Rolling Mean')
plt.plot(Pre_Processed_AAPL['Date'],AAPL_log_std,label='Rolling Std')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')


# ### Split the data into training and test set
# 
#     Training Period: 2015-01-02 - 2020-09-30
# 
#     Testing Period:  2020-10-01 - 2021-02-26

# In[50]:


AAPL_Train_Data=Pre_Processed_AAPL[Pre_Processed_AAPL['Date']<'2020-10-01']
AAPL_Test_Data=Pre_Processed_AAPL[Pre_Processed_AAPL['Date']>='2020-10-01'].reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(AAPL_Train_Data['Date'],AAPL_Train_Data['log Adj Close'],label='Train Data')
plt.plot(AAPL_Test_Data['Date'],AAPL_Test_Data['log Adj Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')


# ## Arima Modeling

# In[51]:


AAPL_Auto_ARIMA_Model=auto_arima(AAPL_Train_Data['log Adj Close'],seasonal=False,
                                error_action='ignore',suppress_warnings=True)
print(AAPL_Auto_ARIMA_Model.summary())


# In[52]:


AAPL_ARIMA_Model=ARIMA(AAPL_Train_Data['log Adj Close'],order=(1,1,0))
AAPL_ARIMA_fit=AAPL_ARIMA_Model.fit()
print(AAPL_ARIMA_fit.summary())


# In[53]:


AAPL_Output=AAPL_ARIMA_fit.forecast(102,alpha=0.05)
AAPL_predictions=np.exp(AAPL_Output[0])


# In[54]:


AAPL_predictions


# In[55]:


AAPL_predictions.shape


# In[56]:


plt.figure(figsize=(10,5))

plt.plot(AAPL_Train_Data['Date'], AAPL_Train_Data['Adj. Close'], label = 'Training')
plt.plot(AAPL_Test_Data['Date'], AAPL_Test_Data['Adj. Close'], label = 'Testing')
plt.plot(AAPL_Test_Data['Date'], AAPL_predictions, label = 'Predictions')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()


# In[57]:


import math
rmse = math.sqrt(mean_squared_error(AAPL_Test_Data['Adj. Close'], AAPL_predictions))
mape = np.mean(np.abs(AAPL_predictions - AAPL_Test_Data['Adj. Close'])/np.abs(AAPL_Test_Data['Adj. Close']))

print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')


# In[58]:


AAPL_Test_Data.shape


# In[59]:


AAPL_Test_Data['Adj. Close']


# In[60]:


AAPL['Adj. Close']


# ### Change Dtype of Date column

# In[61]:


Pre_Processed_AMZN['Date']=pd.to_datetime(Pre_Processed_AAPL['Date'])
Pre_Processed_GOOG['Date']=pd.to_datetime(Pre_Processed_GOOG['Date'])
Pre_Processed_MSFT['Date']=pd.to_datetime(Pre_Processed_MSFT['Date'])
Pre_Processed_TSLA['Date']=pd.to_datetime(Pre_Processed_TSLA['Date'])


# ## Dickey-Fuller Test or adfuller Test

# In[62]:


Test_Stationarity(Pre_Processed_AMZN)


# In[63]:


Test_Stationarity(Pre_Processed_GOOG)


# In[64]:


Test_Stationarity(Pre_Processed_MSFT)


# In[65]:


Test_Stationarity(Pre_Processed_TSLA)


# In[66]:


Pre_Processed_AMZN['log Adj Close']=np.log(Pre_Processed_AMZN['Adj. Close'])
AMZN_log_moving_avg=Pre_Processed_AMZN['log Adj Close'].rolling(12).mean()
AMZN_log_std=Pre_Processed_AMZN['log Adj Close'].rolling(12).std()

plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_AMZN['Date'],AMZN_log_moving_avg,label='Rolling Mean')
plt.plot(Pre_Processed_AMZN['Date'],AMZN_log_std,label='Rolling Std')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')


# In[67]:


Pre_Processed_GOOG['log Adj Close']=np.log(Pre_Processed_GOOG['Adj. Close'])
GOOG_log_moving_avg=Pre_Processed_GOOG['log Adj Close'].rolling(12).mean()
GOOG_log_std=Pre_Processed_GOOG['log Adj Close'].rolling(12).std()

plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_GOOG['Date'],GOOG_log_moving_avg,label='Rolling Mean')
plt.plot(Pre_Processed_GOOG['Date'],GOOG_log_std,label='Rolling Std')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')


# In[68]:


Pre_Processed_MSFT['log Adj Close']=np.log(Pre_Processed_MSFT['Adj. Close'])
MSFT_log_moving_avg=Pre_Processed_MSFT['log Adj Close'].rolling(12).mean()
MSFT_log_std=Pre_Processed_MSFT['log Adj Close'].rolling(12).std()

plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_MSFT['Date'],MSFT_log_moving_avg,label='Rolling Mean')
plt.plot(Pre_Processed_MSFT['Date'],MSFT_log_std,label='Rolling Std')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')


# In[69]:


Pre_Processed_TSLA['log Adj Close']=np.log(Pre_Processed_TSLA['Adj. Close'])
TSLA_log_moving_avg=Pre_Processed_TSLA['log Adj Close'].rolling(12).mean()
TSLA_log_std=Pre_Processed_TSLA['log Adj Close'].rolling(12).std()

plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_TSLA['Date'],TSLA_log_moving_avg,label='Rolling Mean')
plt.plot(Pre_Processed_TSLA['Date'],TSLA_log_std,label='Rolling Std')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')


# ### Split the data into training and test set
# Training Period: 2015-01-02 - 2020-09-30
# 
# Testing Period:  2020-10-01 - 2021-02-26

# In[70]:


AMZN_Train_Data=Pre_Processed_AMZN[Pre_Processed_AMZN['Date']<'2020-10-01']
AMZN_Test_Data=Pre_Processed_AMZN[Pre_Processed_AMZN['Date']>='2020-10-01'].reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(AMZN_Train_Data['Date'],AMZN_Train_Data['log Adj Close'],label='Train Data')
plt.plot(AMZN_Test_Data['Date'],AMZN_Test_Data['log Adj Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')


# In[71]:


GOOG_Train_Data=Pre_Processed_GOOG[Pre_Processed_GOOG['Date']<'2020-10-01']
GOOG_Test_Data=Pre_Processed_GOOG[Pre_Processed_GOOG['Date']>='2020-10-01'].reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(GOOG_Train_Data['Date'],GOOG_Train_Data['log Adj Close'],label='Train Data')
plt.plot(GOOG_Test_Data['Date'],GOOG_Test_Data['log Adj Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')


# In[72]:


MSFT_Train_Data=Pre_Processed_MSFT[Pre_Processed_MSFT['Date']<'2020-10-01']
MSFT_Test_Data=Pre_Processed_MSFT[Pre_Processed_MSFT['Date']>='2020-10-01'].reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(MSFT_Train_Data['Date'],MSFT_Train_Data['log Adj Close'],label='Train Data')
plt.plot(MSFT_Test_Data['Date'],MSFT_Test_Data['log Adj Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')


# In[73]:


TSLA_Train_Data=Pre_Processed_TSLA[Pre_Processed_TSLA['Date']<'2020-10-01']
TSLA_Test_Data=Pre_Processed_TSLA[Pre_Processed_TSLA['Date']>='2020-10-01'].reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(TSLA_Train_Data['Date'],TSLA_Train_Data['log Adj Close'],label='Train Data')
plt.plot(TSLA_Test_Data['Date'],TSLA_Test_Data['log Adj Close'],label='Test Data')
plt.xlabel('Time')
plt.ylabel('log Adj Close')
plt.legend(loc='best')


# ## ARIMA Modeling

# In[74]:


AMZN_Auto_ARIMA_Model=auto_arima(AMZN_Train_Data['log Adj Close'],seasonal=False,
                                 error_action='ignore',supress_warnings=True)

print(AMZN_Auto_ARIMA_Model.summary())


# In[75]:


AMZN_ARIMA_Model=ARIMA(AMZN_Train_Data['log Adj Close'],order=(1,1,0))
AMZN_ARIMA_Model_fit=AMZN_ARIMA_Model.fit()
print(AMZN_ARIMA_Model_fit.summary())


# In[76]:


MSFT_Auto_ARIMA_Model=auto_arima(MSFT_Train_Data['log Adj Close'],seasonal=False,
                                 error_action='ignore',supress_warnings=True)

print(MSFT_Auto_ARIMA_Model.summary())


# In[77]:


MSFT_ARIMA_Model=ARIMA(MSFT_Train_Data['log Adj Close'],order=(1,1,0))
MSFT_ARIMA_Model_fit=MSFT_ARIMA_Model.fit()
print(MSFT_ARIMA_Model_fit.summary())


# In[78]:


GOOG_Auto_ARIMA_Model=auto_arima(GOOG_Train_Data['log Adj Close'],seasonal=False,
                                 error_action='ignore',supress_warnings=True)

print(GOOG_Auto_ARIMA_Model.summary())


# In[79]:


GOOG_ARIMA_Model=ARIMA(GOOG_Train_Data['log Adj Close'],order=(1,1,0))
GOOG_ARIMA_Model_fit=GOOG_ARIMA_Model.fit()
print(GOOG_ARIMA_Model_fit.summary())


# In[80]:


TSLA_Auto_ARIMA_Model=auto_arima(TSLA_Train_Data['log Adj Close'],seasonal=False,
                                 error_action='ignore',supress_warnings=True)

print(TSLA_Auto_ARIMA_Model.summary())


# In[81]:


TSLA_ARIMA_Model=ARIMA(TSLA_Train_Data['log Adj Close'],order=(1,1,0))
TSLA_ARIMA_Model_fit=TSLA_ARIMA_Model.fit()
print(TSLA_ARIMA_Model_fit.summary())


# ## Predicting the closing stock price

# In[82]:


AMZN_output=AMZN_ARIMA_Model_fit.forecast(102,alpha=0.05)
AMZN_predictions=np.exp(AMZN_output[0])

plt.figure(figsize=(10,5))
plt.plot(AMZN_Train_Data['Date'],AMZN_Train_Data['Adj. Close'],label='Training Data')
plt.plot(AMZN_Test_Data['Date'],AMZN_Test_Data['Adj. Close'],label='Testing Data')
plt.plot(AMZN_Test_Data['Date'],AMZN_predictions,label='Predictions')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()


# In[83]:


MSFT_output=MSFT_ARIMA_Model_fit.forecast(102,alpha=0.05)
MSFT_predictions=np.exp(MSFT_output[0])

plt.figure(figsize=(10,5))
plt.plot(MSFT_Train_Data['Date'],MSFT_Train_Data['Adj. Close'],label='Training Data')
plt.plot(MSFT_Test_Data['Date'],MSFT_Test_Data['Adj. Close'],label='Testing Data')
plt.plot(MSFT_Test_Data['Date'],MSFT_predictions,label='Predictions')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()


# In[84]:


GOOG_output=GOOG_ARIMA_Model_fit.forecast(102,alpha=0.05)
GOOG_predictions=np.exp(GOOG_output[0])

plt.figure(figsize=(10,5))
plt.plot(GOOG_Train_Data['Date'],GOOG_Train_Data['Adj. Close'],label='Training Data')
plt.plot(GOOG_Test_Data['Date'],GOOG_Test_Data['Adj. Close'],label='Testing Data')
plt.plot(GOOG_Test_Data['Date'],GOOG_predictions,label='Predictions')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()


# In[85]:


TSLA_output=TSLA_ARIMA_Model_fit.forecast(102,alpha=0.05)
TSLA_predictions=np.exp(TSLA_output[0])

plt.figure(figsize=(10,5))
plt.plot(TSLA_Train_Data['Date'],TSLA_Train_Data['Adj. Close'],label='Training Data')
plt.plot(TSLA_Test_Data['Date'],TSLA_Test_Data['Adj. Close'],label='Testing Data')
plt.plot(TSLA_Test_Data['Date'],TSLA_predictions,label='Predictions')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()


# In[86]:


AMZN_rmse=math.sqrt(mean_squared_error(AMZN_Test_Data['Adj. Close'],AMZN_predictions))
AMZN_mape=np.mean(np.abs(AMZN_predictions-AMZN_Test_Data['Adj. Close'])/np.abs(AMZN_Test_Data['Adj. Close']))
print(f'AMZN_rmse: {AMZN_rmse}')
print(f'AMZN_mape: {AMZN_mape}')


# In[87]:


MSFT_rmse=math.sqrt(mean_squared_error(MSFT_Test_Data['Adj. Close'],MSFT_predictions))
MSFT_mape=np.mean(np.abs(MSFT_predictions-MSFT_Test_Data['Adj. Close'])/np.abs(MSFT_Test_Data['Adj. Close']))
print(f'MSFT_rmse: {MSFT_rmse}')
print(f'MSFT_mape: {MSFT_mape}')


# In[88]:


GOOG_rmse=math.sqrt(mean_squared_error(GOOG_Test_Data['Adj. Close'],GOOG_predictions))
GOOG_mape=np.mean(np.abs(GOOG_predictions-GOOG_Test_Data['Adj. Close'])/np.abs(GOOG_Test_Data['Adj. Close']))
print(f'GOOG_rmse: {GOOG_rmse}')
print(f'GOOG_mape: {GOOG_mape}')


# In[89]:


TSLA_rmse=math.sqrt(mean_squared_error(TSLA_Test_Data['Adj. Close'],TSLA_predictions))
TSLA_mape=np.mean(np.abs(TSLA_predictions-TSLA_Test_Data['Adj. Close'])/np.abs(TSLA_Test_Data['Adj. Close']))
print(f'TSLA_rmse: {TSLA_rmse}')
print(f'TSLA_mape: {TSLA_mape}')


# In[90]:


import tensorflow as tf


# In[91]:


def Dataset(Data,Date):
    
    Train_Data=Data['Adj. Close'][Data['Date']<Date].to_numpy()
    Data_Train=[]
    Data_Train_X=[]
    Data_Train_Y=[]
    for i  in range(0,len(Train_Data),5):
        try:
            Data_Train.append(Train_Data[i:i+5])
        except:
            pass
    
    if len(Data_Train[-1])<5:
        Data_Train.pop(-1)
        
    Data_Train_X=Data_Train[0:-1]
    Data_Train_X=np.array(Data_Train_X)
    Data_Train_X=Data_Train_X.reshape((-1,5,1))
    Data_Train_Y=Data_Train[1:len(Data_Train)]
    Data_Train_Y=np.array(Data_Train_Y)
    Data_Train_Y=Data_Train_Y.reshape((-1,5,1))
    
    
    
    Test_Data=Data['Adj. Close'][Data['Date']>=Date].to_numpy()
    Data_Test=[]
    Data_Test_X=[]
    Data_Test_Y=[]
    for i in range(0,len(Test_Data),5):
        try:
            Data_Test.append(Test_Data[i:i+5])
        except:
            pass
    
    if len(Data_Test[-1])<5:
        Data_Test.pop(-1)
        
    Data_Test_X=Data_Test[0:-1]
    Data_Test_X=np.array(Data_Test_X)
    Data_Test_X=Data_Test_X.reshape((-1,5,1))
    Data_Test_Y=Data_Test[1:len(Data_Test)]
    Data_Test_Y=np.array(Data_Test_Y)
    Data_Test_Y=Data_Test_Y.reshape((-1,5,1))
    
    return Data_Train_X,Data_Train_Y,Data_Test_X,Data_Test_Y


# # Deep Learning Model

# In[92]:


def Model():
    model=tf.keras.models.Sequential([
        tf.keras.layers.LSTM(200,input_shape=(5,1),activation=tf.nn.leaky_relu,return_sequences=True),
        tf.keras.layers.LSTM(200,activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(200,activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(100,activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(50,activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(5,activation=tf.nn.leaky_relu)
        
    ])
    return model


# In[93]:


model=Model()


# In[94]:


tf.keras.utils.plot_model(model,show_shapes=True)


# In[95]:


model.summary()


# ## Custom Learning Rate

# In[96]:


def scheduler(epoch):
    
    if epoch <= 150:
        lrate=(10 ** -5)*(epoch / 150)
    elif epoch <= 400:
        initial_lrate=(10 ** -5)
        k=0.01
        lrate=initial_lrate * math.exp(-k * (epoch - 150))
    else:
        lrate=(10 ** -6)
        
    return lrate


# In[97]:


epochs=[i for i in range(1,1001,1)]
lrate=[scheduler(i) for i in range(1,1001,1)]
plt.plot(epochs,lrate)


# In[98]:


callback=tf.keras.callbacks.LearningRateScheduler(scheduler)


# In[99]:


Pre_Processed_AAPL.head()


# In[100]:


Pre_Processed_AAPL.info()


# ### Split the Data into Training and Test set
#     Training Period: 2015-01-02 - 2020-09-30
# 
#     Testing Period:  2020-10-01 - 2021-02-26

# In[101]:


AAPL_Date='2020-10-01'
AAPL_Train_X,AAPL_Train_Y,AAPL_Test_X,AAPL_Test_Y=
                Dataset(Pre_Processed_AAPL,AAPL_Date)


# In[102]:


AMZN_Date='2020-10-01'
AMZN_Train_X,AMZN_Train_Y,AMZN_Test_X,AMZN_Test_Y=Dataset(Pre_Processed_AMZN,AMZN_Date)


# In[103]:


MSFT_Date='2020-10-01'
MSFT_Train_X,MSFT_Train_Y,MSFT_Test_X,MSFT_Test_Y=Dataset(Pre_Processed_MSFT,MSFT_Date)


# In[104]:


GOOG_Date='2020-10-01'
GOOG_Train_X,GOOG_Train_Y,GOOG_Test_X,GOOG_Test_Y=Dataset(Pre_Processed_GOOG,GOOG_Date)


# In[105]:


TSLA_Date='2020-10-01'
TSLA_Train_X,TSLA_Train_Y,TSLA_Test_X,TSLA_Test_Y=Dataset(Pre_Processed_TSLA,TSLA_Date)


# ## Model Fitting

# ### Apple

# In[106]:


AAPL_Model=Model()


# In[107]:


AAPL_Model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss='mse',
                   metrics=tf.keras.metrics.RootMeanSquaredError())


# In[108]:


AAPL_hist=AAPL_Model.fit(AAPL_Train_X,AAPL_Train_Y,
                         epochs=1000,
                         validation_data=(AAPL_Test_X,AAPL_Test_Y),
                         callbacks=[callback])


# In[109]:


AAPL_history_dict=AAPL_hist.history

AAPL_loss=AAPL_history_dict['loss']
AAPL_root_mean_squared_error=AAPL_history_dict['root_mean_squared_error']
AAPL_val_loss=AAPL_history_dict['val_loss']
AAPL_val_root_mean_squared_error=AAPL_history_dict['val_root_mean_squared_error']

AAPL_epochs=range(1,len(AAPL_loss)+1)


# In[110]:


fig,(ax1,ax2)=plt.subplots(1,2)

fig.set_figheight(5)
fig.set_figwidth(15)
ax1.plot(AAPL_epochs,AAPL_loss,label='Training Loss')
ax1.plot(AAPL_epochs,AAPL_val_loss,label='Validation Loss')
ax1.set(xlabel='Epochs',ylabel='Loss')
ax1.legend()

ax2.plot(AAPL_epochs,AAPL_root_mean_squared_error,label='Training Root Mean Squared Error')
ax2.plot(AAPL_epochs,AAPL_val_root_mean_squared_error,label='Validation Root Mean Squared Error')
ax2.set(xlabel='Epochs',ylabel='Loss')
ax2.legend()

plt.show()


# ## Amazon

# In[111]:


AMZN_Model=Model()


# In[112]:


AMZN_Model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse',metrics=tf.keras.metrics.RootMeanSquaredError())


# In[113]:


AMZN_hist=AMZN_Model.fit(AMZN_Train_X,AMZN_Train_Y,epochs=200,validation_data=(AMZN_Test_X,AMZN_Test_Y),callbacks=[callback])


# In[114]:


AMZN_history_dict=AMZN_hist.history

AMZN_loss=AMZN_history_dict['loss']
AMZN_root_mean_squared_error=AMZN_history_dict['root_mean_squared_error']
AMZN_val_loss=AMZN_history_dict['val_loss']
AMZN_val_root_mean_squared_error=AMZN_history_dict['val_root_mean_squared_error']

AMZN_epochs=range(1,len(AMZN_loss)+1)


# In[115]:


fig,(ax1,ax2)=plt.subplots(1,2)

fig.set_figheight(5)
fig.set_figwidth(15)
ax1.plot(AMZN_epochs,AMZN_loss,label='Training Loss')
ax1.plot(AMZN_epochs,AMZN_val_loss,label='Validation Loss')
ax1.set(xlabel='Epochs',ylabel='Loss')
ax1.legend()

ax2.plot(AMZN_epochs,AMZN_root_mean_squared_error,label='Training Root Mean Squared Error')
ax2.plot(AMZN_epochs,AMZN_val_root_mean_squared_error,label='Validation Root Mean Squared Error')
ax2.set(xlabel='Epochs',ylabel='Loss')
ax2.legend()

plt.show()


# ## Microsoft

# In[116]:


MSFT_Model=Model()


# In[117]:


MSFT_Model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse',metrics=tf.keras.metrics.RootMeanSquaredError())


# In[118]:


MSFT_hist=MSFT_Model.fit(MSFT_Train_X,MSFT_Train_Y,epochs=1000,validation_data=(MSFT_Test_X,MSFT_Test_Y),callbacks=[callback])


# In[119]:


MSFT_history_dict=MSFT_hist.history

MSFT_loss=MSFT_history_dict['loss']
MSFT_root_mean_squared_error=MSFT_history_dict['root_mean_squared_error']
MSFT_val_loss=MSFT_history_dict['val_loss']
MSFT_val_root_mean_squared_error=MSFT_history_dict['val_root_mean_squared_error']

MSFT_epochs=range(1,len(MSFT_loss)+1)


# In[120]:


fig,(ax1,ax2)=plt.subplots(1,2)

fig.set_figheight(5)
fig.set_figwidth(15)
ax1.plot(MSFT_epochs,MSFT_loss,label='Training Loss')
ax1.plot(MSFT_epochs,MSFT_val_loss,label='Validation Loss')
ax1.set(xlabel='Epochs',ylabel='Loss')
ax1.legend()

ax2.plot(MSFT_epochs,MSFT_root_mean_squared_error,label='Training Root Mean Squared Error')
ax2.plot(MSFT_epochs,MSFT_val_root_mean_squared_error,label='Validation Root Mean Squared Error')
ax2.set(xlabel='Epochs',ylabel='Loss')
ax2.legend()

plt.show()


# # GOOGle

# In[121]:


GOOG_Model=Model()


# In[122]:


GOOG_Model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse',metrics=tf.keras.metrics.RootMeanSquaredError())


# In[123]:


GOOG_hist=GOOG_Model.fit(GOOG_Train_X,GOOG_Train_Y,epochs=1000,validation_data=(GOOG_Test_X,GOOG_Test_Y),callbacks=[callback])


# In[124]:


GOOG_history_dict=GOOG_hist.history

GOOG_loss=GOOG_history_dict['loss']
GOOG_root_mean_squared_error=GOOG_history_dict['root_mean_squared_error']
GOOG_val_loss=GOOG_history_dict['val_loss']
GOOG_val_root_mean_squared_error=GOOG_history_dict['val_root_mean_squared_error']

GOOG_epochs=range(1,len(GOOG_loss)+1)


# In[125]:


fig,(ax1,ax2)=plt.subplots(1,2)

fig.set_figheight(5)
fig.set_figwidth(15)
ax1.plot(GOOG_epochs,GOOG_loss,label='Training Loss')
ax1.plot(GOOG_epochs,GOOG_val_loss,label='Validation Loss')
ax1.set(xlabel='Epochs',ylabel='Loss')
ax1.legend()

ax2.plot(GOOG_epochs,GOOG_root_mean_squared_error,label='Training Root Mean Squared Error')
ax2.plot(GOOG_epochs,GOOG_val_root_mean_squared_error,label='Validation Root Mean Squared Error')
ax2.set(xlabel='Epochs',ylabel='Loss')
ax2.legend()

plt.show()


# ## TeSLA

# In[126]:


TSLA_Model=Model()


# In[127]:


TSLA_Model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse',metrics=tf.keras.metrics.RootMeanSquaredError())


# In[128]:


TSLA_hist=TSLA_Model.fit(TSLA_Train_X,TSLA_Train_Y,epochs=200,validation_data=(TSLA_Test_X,TSLA_Test_Y),callbacks=[callback])


# In[129]:


TSLA_history_dict=TSLA_hist.history

TSLA_loss=TSLA_history_dict['loss']
TSLA_root_mean_squared_error=TSLA_history_dict['root_mean_squared_error']
TSLA_val_loss=TSLA_history_dict['val_loss']
TSLA_val_root_mean_squared_error=TSLA_history_dict['val_root_mean_squared_error']

TSLA_epochs=range(1,len(TSLA_loss)+1)


# In[130]:


fig,(ax1,ax2)=plt.subplots(1,2)

fig.set_figheight(5)
fig.set_figwidth(15)
ax1.plot(TSLA_epochs,TSLA_loss,label='Training Loss')
ax1.plot(TSLA_epochs,TSLA_val_loss,label='Validation Loss')
ax1.set(xlabel='Epochs',ylabel='Loss')
ax1.legend()

ax2.plot(TSLA_epochs,TSLA_root_mean_squared_error,label='Training Root Mean Squared Error')
ax2.plot(TSLA_epochs,TSLA_val_root_mean_squared_error,label='Validation Root Mean Squared Error')
ax2.set(xlabel='Epochs',ylabel='Loss')
ax2.legend()

plt.show()


# ## Predictiong the Closing stock price

# In[131]:


AAPL_Prediction=AAPL_Model.predict(AAPL_Test_X)


# In[132]:


plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_AAPL['Date'][Pre_Processed_AAPL['Date']<'2020-10-12'],
         Pre_Processed_AAPL['Adj. Close'][Pre_Processed_AAPL['Date']<'2020-10-12'],
         label='Training')

plt.plot(Pre_Processed_AAPL['Date'][Pre_Processed_AAPL['Date']>='2020-10-09'],
         Pre_Processed_AAPL['Adj. Close'][Pre_Processed_AAPL['Date']>='2020-10-09'],
         label='Testing')

plt.plot(Pre_Processed_AAPL['Date'][Pre_Processed_AAPL['Date']>='2020-10-12'],
         AAPL_Prediction.reshape(-1),
         label='Predictions')

plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='best')


# In[133]:


AAPL_rmse=math.sqrt(mean_squared_error(AAPL_Test_Y.reshape(-1,5),AAPL_Prediction))
AAPL_mape=np.mean(np.abs(AAPL_Prediction-AAPL_Test_Y.reshape(-1,5))/np.abs(AAPL_Test_Y.reshape(-1,5)))

print(f'RMSE: {AAPL_rmse}')
print(f'RMSE: {AAPL_mape}')


# In[134]:


AMZN_Prediction=AMZN_Model.predict(AMZN_Test_X)


# In[135]:


plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_AMZN['Date'][Pre_Processed_AMZN['Date']<'2020-10-12'],
         Pre_Processed_AMZN['Adj. Close'][Pre_Processed_AMZN['Date']<'2020-10-12'],
         label='Training')

plt.plot(Pre_Processed_AMZN['Date'][Pre_Processed_AMZN['Date']>='2020-10-09'],
         Pre_Processed_AMZN['Adj. Close'][Pre_Processed_AMZN['Date']>='2020-10-09'],
         label='Testing')

plt.plot(Pre_Processed_AMZN['Date'][Pre_Processed_AMZN['Date']>='2020-10-12'],
         AMZN_Prediction.reshape(-1),
         label='Predictions')

plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='best')


# In[136]:


AMZN_rmse=math.sqrt(mean_squared_error(AMZN_Test_Y.reshape(-1,5),AMZN_Prediction))
AMZN_mape=np.mean(np.abs(AMZN_Prediction-AMZN_Test_Y.reshape(-1,5))/np.abs(AMZN_Test_Y.reshape(-1,5)))

print(f'RMSE: {AMZN_rmse}')
print(f'RMSE: {AMZN_mape}')


# In[137]:


GOOG_Prediction=GOOG_Model.predict(GOOG_Test_X)


# In[138]:


plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_GOOG['Date'][Pre_Processed_GOOG['Date']<'2020-10-12'],
         Pre_Processed_GOOG['Adj. Close'][Pre_Processed_GOOG['Date']<'2020-10-12'],
         label='Training')

plt.plot(Pre_Processed_GOOG['Date'][Pre_Processed_GOOG['Date']>='2020-10-09'],
         Pre_Processed_GOOG['Adj. Close'][Pre_Processed_GOOG['Date']>='2020-10-09'],
         label='Testing')

plt.plot(Pre_Processed_GOOG['Date'][Pre_Processed_GOOG['Date']>='2020-10-12'],
         GOOG_Prediction.reshape(-1),
         label='Predictions')

plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='best')


# In[139]:


GOOG_rmse=math.sqrt(mean_squared_error(GOOG_Test_Y.reshape(-1,5),GOOG_Prediction))
GOOG_mape=np.mean(np.abs(GOOG_Prediction-GOOG_Test_Y.reshape(-1,5))/np.abs(GOOG_Test_Y.reshape(-1,5)))

print(f'RMSE: {GOOG_rmse}')
print(f'RMSE: {GOOG_mape}')


# In[140]:


MSFT_Prediction=MSFT_Model.predict(MSFT_Test_X)


# In[141]:


plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_MSFT['Date'][Pre_Processed_MSFT['Date']<'2020-10-12'],
         Pre_Processed_MSFT['Adj. Close'][Pre_Processed_MSFT['Date']<'2020-10-12'],
         label='Training')

plt.plot(Pre_Processed_MSFT['Date'][Pre_Processed_MSFT['Date']>='2020-10-09'],
         Pre_Processed_MSFT['Adj. Close'][Pre_Processed_MSFT['Date']>='2020-10-09'],
         label='Testing')

plt.plot(Pre_Processed_MSFT['Date'][Pre_Processed_MSFT['Date']>='2020-10-12'],
         MSFT_Prediction.reshape(-1),
         label='Predictions')

plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='best')


# In[142]:


MSFT_rmse=math.sqrt(mean_squared_error(MSFT_Test_Y.reshape(-1,5),MSFT_Prediction))
MSFT_mape=np.mean(np.abs(MSFT_Prediction-MSFT_Test_Y.reshape(-1,5))/np.abs(MSFT_Test_Y.reshape(-1,5)))

print(f'RMSE: {GOOG_rmse}')
print(f'RMSE: {GOOG_mape}')


# In[143]:


TSLA_Prediction=TSLA_Model.predict(TSLA_Test_X)


# In[144]:


plt.figure(figsize=(10,5))
plt.plot(Pre_Processed_TSLA['Date'][Pre_Processed_TSLA['Date']<'2020-10-12'],
         Pre_Processed_TSLA['Adj. Close'][Pre_Processed_TSLA['Date']<'2020-10-12'],
         label='Training')

plt.plot(Pre_Processed_TSLA['Date'][Pre_Processed_TSLA['Date']>='2020-10-09'],
         Pre_Processed_TSLA['Adj. Close'][Pre_Processed_TSLA['Date']>='2020-10-09'],
         label='Testing')

plt.plot(Pre_Processed_TSLA['Date'][Pre_Processed_TSLA['Date']>='2020-10-12'],
         TSLA_Prediction.reshape(-1),
         label='Predictions')

plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend(loc='best')


# In[145]:


TSLA_rmse=math.sqrt(mean_squared_error(TSLA_Test_Y.reshape(-1,5),TSLA_Prediction))
TSLA_mape=np.mean(np.abs(TSLA_Prediction-TSLA_Test_Y.reshape(-1,5))/np.abs(TSLA_Test_Y.reshape(-1,5)))

print(f'RMSE: {TSLA_rmse}')
print(f'RMSE: {TSLA_mape}')


# In[ ]:


# model.save('keras_model.h5')


# In[ ]:


AAPL_Model.save('karas_model.h5')

