import streamlit as st
import numpy as np
import pandas as pd
from simplified_scrapy import SimplifiedDoc, req
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from cryptory import Cryptory
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import math

def input_data(col1):

    currencies = (
    "Bitcoin",
    "Ether",
    "Litecoin",
    "Dogecoin"
    )

    currency = col1.selectbox("Select Currency", currencies)
    
    if currency == "Bitcoin":
        input_currency = "bitcoin"
    elif currency == "Ether": 
        input_currency = "ethereum"
    elif currency == "Litecoin": 
        input_currency = "litecoin"
    else:
        input_currency = "dogecoin"
    
    return input_currency , currencies

def input_filter(col1):

    models = (
    "FCNN",
    )

    model = col1.selectbox("Select Deep Learning Model", models)

    horizon_slider = col1.slider("Prediction Horizon", 1, 180, 7)

    back_testing = (
    240,
    210,
    180,
    150,
    120,
    90,
    60,
    30,
    )

    back_testing_period = col1.selectbox("Select Backtesting Period", back_testing)

    features = [
    "Market_Cap",
    "Transactions",
    "Amount_Send",
    "Avg_Transaction_Value",
    "Avg_Transaction_Fee",
    "Avg_Block_Time",
    "Avg_Block_Size",
    "Miner_Reward",
    "Mining_Difficulty",
    "Mining_Profitability",
    "Hashrate",
    "Active_Addresses",
    ]

    selected_features = col1.multiselect("Select Blockchain Features", features, features)

    market_feartures = ['Oil_Prices', 'Gold_Prices', 'S&P500_Prices' , 'NASDAQ_Prices' , 'Dow_Jones_Prices' ,'TESLA_Prices' , 'Apple_Prices' , 'NVDIA_Prices' , 'AMD_Prices' , 'Intel_Prices']

    social_feartures = ['Tweet_Volume', 'Google_Trend' , 'Tweets_Subjectivity' , 'Tweets_Polarity']

    selected_market_feartures = col1.multiselect("Select Market Features", market_feartures, market_feartures)

    selected_social_feartures = col1.multiselect("Select Social Features", social_feartures, ['Tweet_Volume' , 'Google_Trend'])
    
    return horizon_slider , selected_features , selected_market_feartures , selected_social_feartures , back_testing_period , model

@st.cache
def collect_data(input  , currencies):

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-price.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Market_Price = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Market_Price[0]= pd.to_datetime(Market_Price[0])
    Market_Price[0] = Market_Price[0].dt.strftime("%m/%d/%y")
    Market_Price[0]= pd.to_datetime(Market_Price[0])
    Market_Price = Market_Price.set_index([0])
    Market_Price = Market_Price.rename(columns={Market_Price.columns[0]:'Market_Price'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Market_Price.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Market_Price.iloc[i] = Market_Price.iloc[i-1]

    # Scraping theData for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-marketcap.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Market_Cap = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Market_Cap[0]= pd.to_datetime(Market_Cap[0])
    Market_Cap[0] = Market_Cap[0].dt.strftime("%m/%d/%y")
    Market_Cap[0]= pd.to_datetime(Market_Cap[0])
    Market_Cap = Market_Cap.set_index([0])
    Market_Cap = Market_Cap.rename(columns={Market_Cap.columns[0]:'Market_Cap'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Market_Cap.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Market_Cap.iloc[i] = Market_Cap.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-transactions.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Transactions = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Transactions[0]= pd.to_datetime(Transactions[0])
    Transactions[0] = Transactions[0].dt.strftime("%m/%d/%y")
    Transactions[0]= pd.to_datetime(Transactions[0])
    Transactions = Transactions.set_index([0])
    Transactions = Transactions.rename(columns={Transactions.columns[0]:'Transactions'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Transactions.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Transactions.iloc[i] = Transactions.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-sentinusd.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Amount_Send = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Amount_Send[0]= pd.to_datetime(Amount_Send[0])
    Amount_Send[0] = Amount_Send[0].dt.strftime("%m/%d/%y")
    Amount_Send[0]= pd.to_datetime(Amount_Send[0])
    Amount_Send = Amount_Send.set_index([0])
    Amount_Send = Amount_Send.rename(columns={Amount_Send.columns[0]:'Amount_Send'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Amount_Send.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Amount_Send.iloc[i] = Amount_Send.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-transactionvalue.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Avg_Transaction_Value = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Avg_Transaction_Value[0]= pd.to_datetime(Avg_Transaction_Value[0])
    Avg_Transaction_Value[0] = Avg_Transaction_Value[0].dt.strftime("%m/%d/%y")
    Avg_Transaction_Value[0]= pd.to_datetime(Avg_Transaction_Value[0])
    Avg_Transaction_Value = Avg_Transaction_Value.set_index([0])
    Avg_Transaction_Value = Avg_Transaction_Value.rename(columns={Avg_Transaction_Value.columns[0]:'Avg_Transaction_Value'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Avg_Transaction_Value.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Avg_Transaction_Value.iloc[i] = Avg_Transaction_Value.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-transactionfees.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Avg_Transaction_Fee = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Avg_Transaction_Fee[0]= pd.to_datetime(Avg_Transaction_Fee[0])
    Avg_Transaction_Fee[0] = Avg_Transaction_Fee[0].dt.strftime("%m/%d/%y")
    Avg_Transaction_Fee[0]= pd.to_datetime(Avg_Transaction_Fee[0])
    Avg_Transaction_Fee = Avg_Transaction_Fee.set_index([0])
    Avg_Transaction_Fee = Avg_Transaction_Fee.rename(columns={Avg_Transaction_Fee.columns[0]:'Avg_Transaction_Fee'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Avg_Transaction_Fee.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Avg_Transaction_Fee.iloc[i] = Avg_Transaction_Fee.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-confirmationtime.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Avg_Block_Time = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Avg_Block_Time[0]= pd.to_datetime(Avg_Block_Time[0])
    Avg_Block_Time[0] = Avg_Block_Time[0].dt.strftime("%m/%d/%y")
    Avg_Block_Time[0]= pd.to_datetime(Avg_Block_Time[0])
    Avg_Block_Time = Avg_Block_Time.set_index([0])
    Avg_Block_Time = Avg_Block_Time.rename(columns={Avg_Block_Time.columns[0]:'Avg_Block_Time'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Avg_Block_Time.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Avg_Block_Time.iloc[i] = Avg_Block_Time.iloc[i-1]
    
    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-size.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Avg_Block_Size = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Avg_Block_Size[0]= pd.to_datetime(Avg_Block_Size[0])
    Avg_Block_Size[0] = Avg_Block_Size[0].dt.strftime("%m/%d/%y")
    Avg_Block_Size[0]= pd.to_datetime(Avg_Block_Size[0])
    Avg_Block_Size = Avg_Block_Size.set_index([0])
    Avg_Block_Size = Avg_Block_Size.rename(columns={Avg_Block_Size.columns[0]:'Avg_Block_Size'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Avg_Block_Size.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Avg_Block_Size.iloc[i] = Avg_Block_Size.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-fee_to_reward.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Miner_Reward = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Miner_Reward[0]= pd.to_datetime(Miner_Reward[0])
    Miner_Reward[0] = Miner_Reward[0].dt.strftime("%m/%d/%y")
    Miner_Reward[0]= pd.to_datetime(Miner_Reward[0])
    Miner_Reward = Miner_Reward.set_index([0])
    Miner_Reward = Miner_Reward.rename(columns={Miner_Reward.columns[0]:'Miner_Reward'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Miner_Reward.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Miner_Reward.iloc[i] = Miner_Reward.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-difficulty.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Mining_Difficulty = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Mining_Difficulty[0]= pd.to_datetime(Mining_Difficulty[0])
    Mining_Difficulty[0] = Mining_Difficulty[0].dt.strftime("%m/%d/%y")
    Mining_Difficulty[0]= pd.to_datetime(Mining_Difficulty[0])
    Mining_Difficulty = Mining_Difficulty.set_index([0])
    Mining_Difficulty = Mining_Difficulty.rename(columns={Mining_Difficulty.columns[0]:'Mining_Difficulty'})    

    # Extracting the Locations of any Possible Null Values
    x = np.where(Mining_Difficulty.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Mining_Difficulty.iloc[i] = Mining_Difficulty.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-mining_profitability.html#1y')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Mining_Profitability = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Mining_Profitability[0]= pd.to_datetime(Mining_Profitability[0])
    Mining_Profitability[0] = Mining_Profitability[0].dt.strftime("%m/%d/%y")
    Mining_Profitability[0]= pd.to_datetime(Mining_Profitability[0])
    Mining_Profitability = Mining_Profitability.set_index([0])
    Mining_Profitability = Mining_Profitability.rename(columns={Mining_Profitability.columns[0]:'Mining_Profitability'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Mining_Profitability.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Mining_Profitability.iloc[i] = Mining_Profitability.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-hashrate.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Hashrate = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Hashrate[0]= pd.to_datetime(Hashrate[0])
    Hashrate[0] = Hashrate[0].dt.strftime("%m/%d/%y")
    Hashrate[0]= pd.to_datetime(Hashrate[0])
    Hashrate = Hashrate.set_index([0])
    Hashrate = Hashrate.rename(columns={Hashrate.columns[0]:'Hashrate'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Hashrate.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Hashrate.iloc[i] = Hashrate.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-activeaddresses.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    Data = [kv.split(',') for kv in js.split('],')]
    Active_Addresses = pd.DataFrame(Data)

    # Correcting Format of Date Columns & Setting as Index
    Active_Addresses[0]= pd.to_datetime(Active_Addresses[0])
    Active_Addresses[0] = Active_Addresses[0].dt.strftime("%m/%d/%y")
    Active_Addresses[0]= pd.to_datetime(Active_Addresses[0])
    Active_Addresses = Active_Addresses.set_index([0])
    Active_Addresses = Active_Addresses.rename(columns={Active_Addresses.columns[0]:'Active_Addresses'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Active_Addresses.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Active_Addresses.iloc[i] = Active_Addresses.iloc[i-1]

    # Combining All of the Data Collected
    Block_Social_Data = pd.concat([Market_Price , Market_Cap , Transactions , Amount_Send , Avg_Transaction_Value , Avg_Transaction_Fee , Avg_Block_Time , Avg_Block_Size ,
                                   Miner_Reward , Mining_Difficulty , Mining_Profitability , Hashrate , Active_Addresses] , axis=1)

    # Dropping Any Rows with Missing Values
    Block_Social_Data = Block_Social_Data.dropna()

    # Converting Data from Object to Float
    Block_Social_Data["Avg_Block_Size"] = pd.to_numeric(Block_Social_Data["Avg_Block_Size"], downcast="float")
    Block_Social_Data["Miner_Reward"] = pd.to_numeric(Block_Social_Data["Miner_Reward"], downcast="float")
    Block_Social_Data["Mining_Difficulty"] = pd.to_numeric(Block_Social_Data["Mining_Difficulty"], downcast="float")
    Block_Social_Data["Mining_Profitability"] = pd.to_numeric(Block_Social_Data["Mining_Profitability"], downcast="float")
    Block_Social_Data["Active_Addresses"] = pd.to_numeric(Block_Social_Data["Active_Addresses"], downcast="float")
    Block_Social_Data["Hashrate"] = pd.to_numeric(Block_Social_Data["Hashrate"], downcast="float")
    Block_Social_Data["Avg_Block_Time"] = pd.to_numeric(Block_Social_Data["Avg_Block_Time"], downcast="float")
    Block_Social_Data["Avg_Transaction_Fee"] = pd.to_numeric(Block_Social_Data["Avg_Transaction_Fee"], downcast="float")
    Block_Social_Data["Avg_Transaction_Value"] = pd.to_numeric(Block_Social_Data["Avg_Transaction_Value"], downcast="float")
    Block_Social_Data["Amount_Send"] = pd.to_numeric(Block_Social_Data["Amount_Send"], downcast="float")
    Block_Social_Data["Transactions"] = pd.to_numeric(Block_Social_Data["Transactions"], downcast="float")
    Block_Social_Data["Market_Cap"] = pd.to_numeric(Block_Social_Data["Market_Cap"], downcast="float")
    Block_Social_Data["Market_Price"] = pd.to_numeric(Block_Social_Data["Market_Price"], downcast="float")
    Block_Social_Data.index = Block_Social_Data.index.strftime('%m/%d/%Y')

    # Collecting Martket Data
    tickers = ['CL=F', 'GC=F', '^GSPC' , '^IXIC' , '%5EDJI' ,'TSLA' , 'AAPL' , 'NVDA' , 'AMD' , 'INTC']

    Market_Data = yf.download(tickers, start='2014-09-30', progress=False)['Close']

    # Correcting Format of Date Columns & Setting as Index
    Market_Data.index = Market_Data.index.strftime("%m/%d/%y")
    Market_Data.index = pd.to_datetime(Market_Data.index)
    Market_Data = Market_Data.rename(columns={Market_Data.columns[0]:'Gold_Prices' , Market_Data.columns[1]:'Oil_Prices' ,
                                          Market_Data.columns[2]:'S&P500_Prices' , Market_Data.columns[3]:'NASDAQ_Prices',
                                          Market_Data.columns[4]:'Dow_Jones_Prices' , Market_Data.columns[5]:'TESLA_Prices',
                                          Market_Data.columns[6]:'Apple_Prices' , Market_Data.columns[7]:'NVDIA_Prices',
                                          Market_Data.columns[8]:'AMD_Prices' , Market_Data.columns[9]:'Intel_Prices'})
    
    # Combining all Features into Single Dataframe with Common DateTime Index Same as BTC Data
    Market_Data = pd.concat([Market_Price, Market_Data] ,axis=1)
    Market_Data = Market_Data.drop(['Market_Price'], axis = 1)
    Market_Data = Market_Data['2014-09-30':]

    # Back & Front Filling the Missing Weekend Data 
    Market_Data['Oil_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['Oil_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['Gold_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['Gold_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['S&P500_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['S&P500_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['NASDAQ_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['NASDAQ_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['Dow_Jones_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['Dow_Jones_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['TESLA_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['TESLA_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['Apple_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['Apple_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['NVDIA_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['NVDIA_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['AMD_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['AMD_Prices'].fillna(method='ffill', inplace=True)
    Market_Data['Intel_Prices'].fillna(method='bfill', inplace=True)
    Market_Data['Intel_Prices'].fillna(method='ffill', inplace=True)
    Market_Data.index = Market_Data.index.strftime('%m/%d/%Y')

    # Collecting Social Data

    # Scraping the Data From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-tweets.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Tweet_Volume = pd.DataFrame(data)

    # Correcting Format of Date Columns & Setting as Index
    Tweet_Volume[0]= pd.to_datetime(Tweet_Volume[0])
    Tweet_Volume[0] = Tweet_Volume[0].dt.strftime("%m/%d/%y")
    Tweet_Volume[0]= pd.to_datetime(Tweet_Volume[0])
    Tweet_Volume = Tweet_Volume.set_index([0])
    Tweet_Volume = Tweet_Volume.rename(columns={Tweet_Volume.columns[0]:'Tweet_Volume'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Tweet_Volume.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Tweet_Volume.iloc[i] = Tweet_Volume.iloc[i-1]

    # Getting Google Trend Data
    my_cryptory = Cryptory(from_date="2014-09-30")
    Google_Trend_Data = my_cryptory.get_google_trends(kw_list=[currencies[0]])

    # Correcting Format of Date Columns & Setting as Index
    Google_Trend_Data = Google_Trend_Data.set_index(['date'])
    Google_Trend_Data = Google_Trend_Data.rename(columns={Google_Trend_Data.columns[0]:'Google_Trend'})

    # Combining All of the Data Collected
    Social_Data = pd.concat([Google_Trend_Data , Tweet_Volume] , axis=1)

    # Dropping Any Rows with Missing Values
    Social_Data = Social_Data.dropna()

    # Converting Data from Object to Float
    Social_Data["Google_Trend"] = pd.to_numeric(Social_Data["Google_Trend"], downcast="float")
    Social_Data["Tweet_Volume"] = pd.to_numeric(Social_Data["Tweet_Volume"], downcast="float")
    Social_Data.index = Social_Data.index.strftime('%m/%d/%Y')

    Combined_Data = pd.concat([Block_Social_Data , Market_Data , Social_Data] , axis = 1)

    # Dropping Any Rows with Missing Values
    Combined_Data = Combined_Data.dropna()

    return Block_Social_Data , Market_Data , Social_Data

def filter_data(Block_Social_Data , Market_Data , Social_Data , selected_features , selected_market_feartures , selected_social_feartures):

    Block_Social_Data = Block_Social_Data[['Market_Price'] + selected_features]
    Market_Data = Market_Data[selected_market_feartures]
    Social_Data = Social_Data[selected_social_feartures]
    Combined_Data = pd.concat([Block_Social_Data , Market_Data , Social_Data] , axis = 1)
    Combined_Data = Combined_Data.dropna()
    return Combined_Data

def pre_processing(prediction_days , Combined_Data  , back_testing_period):

    prediction_days = prediction_days
    Combined_Data['Prediction'] = Combined_Data[['Market_Price']].shift(-prediction_days)

    forecast_Data = Combined_Data[Combined_Data['Prediction'].isna()]

    # Dropping the Resulting NA Rows for the New Column
    Combined_Data = Combined_Data.dropna()

    # Generating the Target Variable for Classification Problem
    Combined_Data['Price_Increase'] = Combined_Data['Prediction'].gt(Combined_Data['Market_Price'])*1

    # Generating the Target Variable for Investment Return Regression Problem
    Combined_Data['Investment_Return'] = round((Combined_Data['Prediction'] - Combined_Data['Market_Price'])/Combined_Data['Market_Price']*100,1)

    # Creating Training Data Data Frame
    test_1 = pd.to_datetime(Combined_Data.index[-1]) - relativedelta(days=back_testing_period)
    enddate1 = pd.to_datetime(test_1).date().strftime('%m/%d/%Y')
    train_data = Combined_Data[:enddate1]

    # Creating Testing Data Data Frame
    test_data = Combined_Data[enddate1:]

    # Creating the Train/Test Split Graph
    fig, ax = plt.subplots(1, figsize=(12, 4))
    ax.plot(train_data['Prediction'], label='Training', linewidth=2 , color = "green")
    ax.plot(test_data['Prediction'],  label='Validation', linewidth=2 , color = "blue" )
    ax.set_ylabel('Market Price', fontsize=14)
    ax.set_xlabel('Time', fontsize=14)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.legend(['Training', 'Validation'], loc = 'best')
    plt.axvline(x=enddate1,color='black')
    st.title("Visualization of Train/Validation Split")
    st.title("")
    st.pyplot(plt)

    # Splitting the Features & Target Variables for Training Data
    x_train = np.array(train_data.drop(['Prediction','Price_Increase' , 'Investment_Return'],1))
    y_train = np.array(train_data[['Prediction','Price_Increase' , 'Investment_Return']])

    # Reshaping the Target Variables for Traning Set
    y_train_1 = y_train[:,0].reshape(-1,1)
    y_train_2 = y_train[:,1].reshape(-1,1)
    y_train_3 = y_train[:,2].reshape(-1,1)

    # Splitting the Features & Target Variables for Test Data
    x_test = np.array(test_data.drop(['Prediction','Price_Increase' , 'Investment_Return'],1))
    x_forecast = np.array(forecast_Data.drop(['Prediction'],1))
    y_test_1 = np.array(test_data['Prediction'])
    y_test_2 = np.array(test_data['Price_Increase'])
    y_test_3 = np.array(test_data['Investment_Return'])

    # Reshaping the Target Variables
    y_test_1 = y_test_1.reshape(-1,1)
    y_test_2 = y_test_2.reshape(-1,1)
    y_test_3 = y_test_3.reshape(-1,1)

    # Numeric Scaling for Features
    f_sc = MinMaxScaler()
    x_train = f_sc.fit_transform(x_train)
    x_test = f_sc.transform(x_test)
    x_forecast = f_sc.transform(x_forecast)

    # Numeric Scaling for Target Features
    sc_1 = MinMaxScaler()
    sc_2 = MinMaxScaler()
    y_train_1 = sc_1.fit_transform(y_train_1).ravel().reshape(-1,1)
    y_test_1 = sc_1.transform(y_test_1).ravel().reshape(-1,1)
    y_train_3 = sc_2.fit_transform(y_train_3).ravel().reshape(-1,1)
    y_test_3 = sc_2.transform(y_test_3).ravel().reshape(-1,1)

    return x_train , x_test , y_train_1 , y_train_2 , y_train_3 , y_test_1 , y_test_2 , y_test_3 , test_data , sc_1 , x_forecast , train_data , forecast_Data

def train_model_FCNN(x_train,x_test,y_train_1,y_test_1  , test_data  , sc_1 , x_forecast , train_data , forecast_Data):

    # Determining Number of Nuerons in Input Layer (Number of Features)
    n_features = x_train.shape[1]

    def create_model(optimizer = 'adam' , activation = 'relu' , dropout_rate = 0.0 , random_state = 777 ):
        model = Sequential()
        model.add(Dense(64, activation = activation, input_shape = (n_features,)))
        model.add(Dropout(rate = dropout_rate))
        model.add(Dense(1 ,activation="linear"))
        model.compile(optimizer = optimizer , loss = 'mse', metrics = ['mse'])
        return model
    
    # Defining the Grid Search Parameters
    batch_size = [32, 64]
    epochs = [10, 20]
    optimizer = ['RMSprop' ,'Adam']
    activation = ['relu', 'tanh']
    dropout_rate = [0.4, 0.8]
    param_grid = dict(batch_size = batch_size, epochs  = epochs ,activation = activation, optimizer = optimizer , dropout_rate = dropout_rate)

    # Building the Grid Search Model
    model = KerasRegressor(build_fn = create_model, verbose = 0)
    grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3 , scoring = 'neg_mean_squared_error')
    grid_result_train = grid.fit(x_train, y_train_1)

    # Summarizing the Best Resulting Model
    st.write("Model: %s" % (grid_result_train.best_params_))

    # Plotting Results of Gridsearch Validation Sets
    fig, ax = plt.subplots(1, figsize=(13, 7))
    test_scores = grid.cv_results_['mean_test_score']
    split0_test_score = grid.cv_results_['split0_test_score']
    split1_test_score = grid.cv_results_['split1_test_score']
    split2_test_score = grid.cv_results_['split2_test_score']
    plt.plot(test_scores, label='Mean')
    plt.plot(split0_test_score, label='CV 1')
    plt.plot(split1_test_score, label='CV 2')
    plt.plot(split2_test_score, label='CV 3')
    plt.legend(loc='best')
    st.title("Cross Validation Mean Scores")
    st.title("")
    st.pyplot(fig)

    st.write(grid.best_score_)

    st.write(grid.best_params_)

    #st.write(model.summary())

    Ypred_inverse = grid.predict(x_train).reshape(-1,1)
    Ypred_inverse = sc_1.inverse_transform(Ypred_inverse)
    y_actual = sc_1.inverse_transform(y_train_1)

    # creating a DF of the predicted prices
    Ypred_inverse = pd.DataFrame(Ypred_inverse, index=train_data.index)
    y_actual = pd.DataFrame(y_actual, index=train_data.index)

    fig, ax = plt.subplots(1, figsize=(13, 7))
    plt.plot(y_actual, color = 'red', label = 'Actual Price');
    plt.plot(Ypred_inverse, color = 'blue', label = 'Predicted Price');
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.legend(loc='best', fontsize=16);
    st.title("")
    st.title("Visualizing Training Accuracy")
    st.title("")
    st.pyplot(plt)
    Adj_r2 = 1 - (1-r2_score(Ypred_inverse, y_actual)) * (len(y_actual)-1)/(len(y_actual) - x_test.shape[1] - 1)
    rmse = math.sqrt(mean_squared_error(y_actual, Ypred_inverse))
    mse = mean_absolute_error(y_actual, Ypred_inverse)
    ev = explained_variance_score(y_actual, Ypred_inverse)
    st.title("")
    st.write('RMSE Score: {}'.format(round(rmse, 2)))
    st.write('MASE Score: {}'.format(round(mse, 2)))
    st.write('Explained Varaince Score: {}'.format(round(ev, 2)))
    st.write('Adj R Squared Score: {}'.format(round(Adj_r2, 2)))

    Ypred_inverse = grid.predict(x_test).reshape(-1,1)
    Ypred_inverse_test = sc_1.inverse_transform(Ypred_inverse)
    y_actual = sc_1.inverse_transform(y_test_1)
    Ypred_inverse = grid.predict(x_forecast).reshape(-1,1)
    Ypred_inverse = sc_1.inverse_transform(Ypred_inverse)

    # creating a DF of the predicted prices
    Ypred_inverse_test = pd.DataFrame(Ypred_inverse_test, index=test_data.index , columns = ["Prediction"])
    y_actual = pd.DataFrame(y_actual, index=test_data.index)
    Ypred_inverse_forecast = pd.DataFrame(Ypred_inverse, index=forecast_Data.index , columns = ["Prediction"])
    Ypred_inverse = pd.concat([Ypred_inverse_test,Ypred_inverse_forecast] , axis = 0)
  
    fig, ax = plt.subplots(1, figsize=(13, 7))
    plt.plot(y_actual, color = 'red', label = 'Actual Price');
    plt.plot(Ypred_inverse, color = 'blue', label = 'Predicted Price');
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.legend(loc='best', fontsize=16);
    st.title("")
    st.title("Visualizing Testing Accuracy")
    plt.axvline(x=len(y_actual), color='black'  , label= 'Train/Validation Split')
    st.title("")
    st.pyplot(plt)
    Adj_r2 = 1 - (1-r2_score(Ypred_inverse_test, y_actual)) * (len(y_actual)-1)/(len(y_actual) - x_test.shape[1] - 1)
    rmse = math.sqrt(mean_squared_error(y_actual, Ypred_inverse_test))
    mse = mean_absolute_error(y_actual, Ypred_inverse_test)
    ev = explained_variance_score(y_actual, Ypred_inverse_test)
    st.title("")
    st.write('RMSE Score: {}'.format(round(rmse, 2)))
    st.write('MASE Score: {}'.format(round(mse, 2)))
    st.write('Explained Varaince Score: {}'.format(round(ev, 2)))
    st.write('Adj R Squared Score: {}'.format(round(Adj_r2, 2)))
    st.title("")
    st.title("Future Price Predictions")
    st.title("")

    st.write(Ypred_inverse_forecast)

    Ypred_inverse_forecast['Day_Before'] = Ypred_inverse_forecast.shift(periods=1)
    Ypred_inverse_forecast['Price_Increase'] = Ypred_inverse_forecast['Prediction'].gt(Ypred_inverse_forecast['Day_Before'])*1
    Ypred_inverse_forecast['Investment_Return'] = round((Ypred_inverse_forecast['Prediction'] - Ypred_inverse_forecast['Day_Before'])/Ypred_inverse_forecast['Day_Before']*100,1)
    Ypred_inverse_forecast['cum_sum'] = Ypred_inverse_forecast['Investment_Return'].cumsum()
    Ypred_inverse_forecast['positive_percent_change'] = Ypred_inverse_forecast['cum_sum'] > 0
    Ypred_inverse_forecast = Ypred_inverse_forecast.dropna()

    st.title("")
    st.title("Predicted Cumulative Returns")
    st.title("")

    fig, ax = plt.subplots(figsize=(10,5))
    plt.subplots_adjust(top = 1, bottom = 0)
    ax = Ypred_inverse_forecast['cum_sum'].plot(kind ='bar', color = Ypred_inverse_forecast.positive_percent_change.map({True: 'g', False: 'r'}) ,label = "")
    for p in ax.patches:
        ax.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    ax.set_ylim([-1*Ypred_inverse_forecast.cum_sum.max()*1.15, Ypred_inverse_forecast.cum_sum.max()*1.15])
    ax.set_xlabel('')
    ax.set_ylabel('')
    st.pyplot(fig)



def show_Deep_Learning_Model():

    col1 = st.sidebar


    input , currencies = input_data(col1)

    horizon_slider , selected_features , selected_market_feartures , selected_social_feartures , back_testing_period , model = input_filter(col1)
  
    Block_Social_Data , Market_Data , Social_Data = collect_data(input , currencies)

    df = filter_data(Block_Social_Data , Market_Data , Social_Data , selected_features , selected_market_feartures , selected_social_feartures )
   
    if col1.checkbox(label = "Show Original Data"):
        st.title("Original Data")
        st.title("")
        st.write(df)

    x_train , x_test , y_train_1 , y_train_2 , y_train_3 , y_test_1 , y_test_2 , y_test_3 , test_data , sc_1 , x_forecast , train_data , forecast_Data = pre_processing(horizon_slider , df , back_testing_period)

    forecast_Data.index  = pd.to_datetime(forecast_Data.index) + pd.DateOffset(days=horizon_slider)
    forecast_Data.index = forecast_Data.index.strftime('%m/%d/%Y')
    train_data.index  = pd.to_datetime(train_data.index) + pd.DateOffset(days=horizon_slider)
    train_data.index = train_data.index.strftime('%m/%d/%Y')
    test_data.index  = pd.to_datetime(test_data.index) + pd.DateOffset(days=horizon_slider)
    test_data.index = test_data.index.strftime('%m/%d/%Y')

    if col1.checkbox(label = "Show Training & Testing Data"):
        st.title("Processed Tarining Data")
        st.title("")
        st.write(x_train)
        st.title("Processed Testing Data")
        st.title("")
        st.write(x_test)
    
    train_model_FCNN(x_train , x_test , y_train_1 , y_test_1  , test_data  , sc_1 , x_forecast , train_data , forecast_Data)

    
   



   
