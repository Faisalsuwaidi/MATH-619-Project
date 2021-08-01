import streamlit as st
import numpy as np
import pandas as pd
from simplified_scrapy import SimplifiedDoc, req
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def input_data(col1):

    currencies = (
    "Bitcoin",
    "Ether",
    "Litecoin",
    "Dogecoin",
    )

    currency = col1.selectbox("Select Feature", currencies)
    
    if currency == "Bitcoin":
        input_currency = "bitcoin"
    elif currency == "Ether": 
        input_currency = "ethereum"
    elif currency == "Litecoin": 
        input_currency = "litecoin"
    else: 
        input_currency = "dogecoin"
    
    return input_currency  

def collect_data(input):

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-price.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Market_Price = pd.DataFrame(data)

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
    
    Market_Price = Market_Price.loc['2014-09-30':]

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

    Combined_Data = pd.concat([Market_Price , Market_Data] , axis = 1)

    # Dropping Any Rows with Missing Values
    Combined_Data = Combined_Data.dropna()

    # Converting Data from Object to Float
    Combined_Data["Market_Price"] = pd.to_numeric(Combined_Data["Market_Price"], downcast="float")
    Combined_Data["Oil_Prices"] = pd.to_numeric(Combined_Data["Oil_Prices"], downcast="float")
    Combined_Data["Gold_Prices"] = pd.to_numeric(Combined_Data["Gold_Prices"], downcast="float")
    Combined_Data["S&P500_Prices"] = pd.to_numeric(Combined_Data["S&P500_Prices"], downcast="float")
    Combined_Data["NASDAQ_Prices"] = pd.to_numeric(Combined_Data["NASDAQ_Prices"], downcast="float")
    Combined_Data["Dow_Jones_Prices"] = pd.to_numeric(Combined_Data["Dow_Jones_Prices"], downcast="float")
    Combined_Data["TESLA_Prices"] = pd.to_numeric(Combined_Data["TESLA_Prices"], downcast="float")
    Combined_Data["Apple_Prices"] = pd.to_numeric(Combined_Data["Apple_Prices"], downcast="float")
    Combined_Data["NVDIA_Prices"] = pd.to_numeric(Combined_Data["NVDIA_Prices"], downcast="float")
    Combined_Data["AMD_Prices"] = pd.to_numeric(Combined_Data["AMD_Prices"], downcast="float")
    Combined_Data["Intel_Prices"] = pd.to_numeric(Combined_Data["Intel_Prices"], downcast="float")
    Combined_Data.index = Combined_Data.index.strftime('%m/%d/%Y')

     # Dropping Any Rows with Missing Values
    Combined_Data = Combined_Data.dropna()

    Return_Data = Combined_Data.pct_change(1).dropna()
    Cum_Return_Data  = (Return_Data + 1).cumprod() 
    Cum_Return_Data = Cum_Return_Data.drop(['Market_Price'], axis = 1)

    return Combined_Data , Return_Data , Cum_Return_Data

def show_Market_Commodity_Data_EDA():

    col1 = st.sidebar

    input = input_data(col1)
    df , Return_Data , Cum_Return_Data  = collect_data(input)

    if st.sidebar.checkbox(label = "Display Market Data"):
        st.title("Market Data")
        st.write(df)
    
    st.title("Market Data Plots")
    plotly_figure = px.line(data_frame = df, x = df.index , y = df.columns, log_y = True )
    st.plotly_chart(plotly_figure)

    st.title("Market Data Correlation Analysis")
    cor_feature = 'Market_Price'
    heatmap = sns.heatmap(df.corr()[[cor_feature]].sort_values(by=cor_feature, ascending=False), vmin=-1, vmax=1,linewidths = 1, robust = True ,annot=True, cmap='Blues')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.title('')
    fig , axs = plt.subplots(2 , 2 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
    axs[0,0].plot(Return_Data["Oil_Prices"] , c = 'r')
    axs[0,0].set_title('Oil')
    axs[0,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0,0].set_ylim([-0.6,0.6])
    axs[0,1].plot(Return_Data["Gold_Prices"] , c = 'g')
    axs[0,1].set_title('Gold')
    axs[0,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0,1].set_ylim([-0.6,0.6])
    axs[1,0].plot(Return_Data["S&P500_Prices"] , c = 'y')
    axs[1,0].set_title('S&P500')
    axs[1,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1,0].set_ylim([-0.6,0.6])
    axs[1,1].plot(Return_Data["NASDAQ_Prices"] , c = 'b')
    axs[1,1].set_title('NASDAQ')
    axs[1,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1,1].set_ylim([-0.6,0.6])
    fig.suptitle('Daily Returns Volatlity Plots',fontweight ="bold",  fontsize=30)
    st.pyplot(fig)

    st.title("Growth of $1 investment Plots")
    plotly_figure = px.line(data_frame = Cum_Return_Data, x = Cum_Return_Data.index , y = Cum_Return_Data.columns, log_y = False )
    st.plotly_chart(plotly_figure)
    