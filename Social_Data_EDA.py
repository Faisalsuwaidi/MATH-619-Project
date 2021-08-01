import streamlit as st
import numpy as np
import pandas as pd
from simplified_scrapy import SimplifiedDoc, req
import plotly.express as px
from cryptory import Cryptory
import seaborn as sns

def input_data(col1):

    currencies = [
    "Bitcoin",
    "Ether",
    "Litecoin",
    "Dogecoin",
    ]

    currency = col1.selectbox("Select Feature", currencies)
    
    if currency == "Bitcoin":
        input_currency = "bitcoin"
    elif currency == "Ether": 
        input_currency = "ethereum"
    elif currency == "Litecoin": 
        input_currency = "litecoin"
    else: 
        input_currency = "dogecoin"
    
    return input_currency , currencies

def collect_data(input , currencies):

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
    Social_Data = pd.concat([Market_Price , Google_Trend_Data , Tweet_Volume] , axis=1)

    # Dropping Any Rows with Missing Values
    Social_Data = Social_Data.dropna()

    # Converting Data from Object to Float
    Social_Data["Market_Price"] = pd.to_numeric(Social_Data["Market_Price"], downcast="float")
    Social_Data["Google_Trend"] = pd.to_numeric(Social_Data["Google_Trend"], downcast="float")
    Social_Data["Tweet_Volume"] = pd.to_numeric(Social_Data["Tweet_Volume"], downcast="float")
    Social_Data.index = Social_Data.index.strftime('%m/%d/%Y')

    Return_Data = Social_Data.pct_change(1).dropna()

    return Social_Data , Return_Data
   
def show_Social_Data_EDA():

    col1 = st.sidebar

    input , currencies = input_data(col1)
    df , Return_Data = collect_data(input , currencies)

    if st.sidebar.checkbox(label = "Display Social Data"):
       st.title("Social Data")
       st.write(df)

    st.title("Social Data Plots")
    plotly_figure = px.line(data_frame = df, x = df.index , y = ['Market_Price' , 'Google_Trend' , 'Tweet_Volume'], log_y = True )
    st.plotly_chart(plotly_figure)

    st.title("Market Price Volatlity Plot")
    plotly_figure = px.line(data_frame = Return_Data, x = Return_Data.index , y = ['Market_Price'], log_y = False )
    st.plotly_chart(plotly_figure)

    st.title("Google Trend Volatlity Plot")
    plotly_figure = px.line(data_frame = Return_Data, x = Return_Data.index , y = ['Google_Trend'], log_y = False , color_discrete_map={"Google_Trend": "goldenrod"})
                 
    st.plotly_chart(plotly_figure)

    st.title("Tweet Volume Voliality Plot")
    plotly_figure = px.line(data_frame = Return_Data, x = Return_Data.index , y = ['Tweet_Volume'], log_y = False , color_discrete_map={"Tweet_Volume": "#EF553B"})
    st.plotly_chart(plotly_figure)

    st.title("Scoial Data Correlation Analysis")
    cor_feature = 'Market_Price'
    heatmap = sns.heatmap(df.corr()[[cor_feature]].sort_values(by=cor_feature, ascending=False), vmin=-1, vmax=1,linewidths = 1, robust = True ,annot=True, cmap='Blues')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    