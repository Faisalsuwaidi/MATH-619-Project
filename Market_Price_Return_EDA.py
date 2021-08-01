import streamlit as st
import numpy as np
import pandas as pd
from simplified_scrapy import SimplifiedDoc, req
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def input_data():

    currencies = [
    "Bitcoin",
    "Ether",
    "Litecoin",
    "Dogecoin"
    ]
    
    return currencies

def collect_data(input):

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/bitcoin-price.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Bitcoin = pd.DataFrame(data)

    # Correcting Format of Date Columns & Setting as Index
    Bitcoin[0]= pd.to_datetime(Bitcoin[0])
    Bitcoin[0] = Bitcoin[0].dt.strftime("%m/%d/%y")
    Bitcoin[0]= pd.to_datetime(Bitcoin[0])
    Bitcoin = Bitcoin.set_index([0])
    Bitcoin = Bitcoin.rename(columns={Bitcoin.columns[0]:'Bitcoin'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Bitcoin.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Bitcoin.iloc[i] = Bitcoin.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/ethereum-price.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Ether = pd.DataFrame(data)

    # Correcting Format of Date Columns & Setting as Index
    Ether[0]= pd.to_datetime(Ether[0])
    Ether[0] = Ether[0].dt.strftime("%m/%d/%y")
    Ether[0]= pd.to_datetime(Ether[0])
    Ether = Ether.set_index([0])
    Ether = Ether.rename(columns={Ether.columns[0]:'Ether'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Ether.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Ether.iloc[i] = Ether.iloc[i-1]

    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/litecoin-price.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Litecoin = pd.DataFrame(data)

    # Correcting Format of Date Columns & Setting as Index
    Litecoin[0]= pd.to_datetime(Litecoin[0])
    Litecoin[0] = Litecoin[0].dt.strftime("%m/%d/%y")
    Litecoin[0]= pd.to_datetime(Litecoin[0])
    Litecoin = Litecoin.set_index([0])
    Litecoin = Litecoin.rename(columns={Litecoin.columns[0]:'Litecoin'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Litecoin.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Litecoin.iloc[i] = Litecoin.iloc[i-1]
    
    # Scraping the Data for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/dogecoin-price.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Dogecoin = pd.DataFrame(data)

    # Correcting Format of Date Columns & Setting as Index
    Dogecoin[0]= pd.to_datetime(Dogecoin[0])
    Dogecoin[0] = Dogecoin[0].dt.strftime("%m/%d/%y")
    Dogecoin[0]= pd.to_datetime(Dogecoin[0])
    Dogecoin = Dogecoin.set_index([0])
    Dogecoin = Dogecoin.rename(columns={Dogecoin.columns[0]:'Dogecoin'})

    # Extracting the Locations of any Possible Null Values
    x = np.where(Dogecoin.applymap(lambda x: x == 'null'))
    x = x[0]

    # Backfilling any Null Values Automatically 
    for i in x:
        Dogecoin.iloc[i] = Dogecoin.iloc[i-1]
    
    # Combining All of the Data Collected
    Price_Data = pd.concat([Bitcoin , Ether , Litecoin , Dogecoin ] , axis=1)

    # Dropping Any Rows with Missing Values
    Price_Data = Price_Data.dropna()

    # Converting Data from Object to Float
    Price_Data["Bitcoin"] = pd.to_numeric(Price_Data["Bitcoin"], downcast="float")
    Price_Data["Ether"] = pd.to_numeric(Price_Data["Ether"], downcast="float")
    Price_Data["Litecoin"] = pd.to_numeric(Price_Data["Litecoin"], downcast="float")
    Price_Data["Dogecoin"] = pd.to_numeric(Price_Data["Dogecoin"], downcast="float")
    Price_Data.index = Price_Data.index.strftime('%m/%d/%Y')

    Return_Data = Price_Data.pct_change(1).dropna()

    daily_cum_returns = (Return_Data + 1).cumprod() 

    currency_price_selection = Price_Data[input]

    currency_return_selection = Return_Data[input]

    currency_cum_returns_selection = daily_cum_returns[input]

    return Price_Data , currency_return_selection , currency_price_selection , currency_cum_returns_selection

def show_Market_Price_Return_EDA():

    input = input_data()
    df , currency_return_selection , currency_price_selection , currency_cum_returns_selection = collect_data(input)

    if st.sidebar.checkbox(label = "Display Market Price Data"):
        st.title("Market Price Data")
        st.write(currency_price_selection)
    if st.sidebar.checkbox(label = "Display Return Data"):
        st.title("Return Data")
        st.write(currency_return_selection)

    st.title('')
    fig , axs = plt.subplots(2 , 2 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
    axs[0,0].plot(currency_price_selection["Bitcoin"] , c = 'r')
    axs[0,0].set_title('BTC')
    axs[0,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0,1].plot(currency_price_selection["Ether"] , c = 'g')
    axs[0,1].set_title('ETH')
    axs[0,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1,0].plot(currency_price_selection["Litecoin"] , c = 'y')
    axs[1,0].set_title('LTC')
    axs[1,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1,1].plot(currency_price_selection["Dogecoin"] , c = 'b')
    axs[1,1].set_title('DGC')
    axs[1,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    fig.suptitle('Market Price Plots',fontweight ="bold" ,  fontsize=30)
    st.pyplot(fig)

    st.title('')
    fig , axs = plt.subplots(2 , 2 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
    axs[0,0].plot(currency_return_selection["Bitcoin"] , c = 'r')
    axs[0,0].set_title('BTC')
    axs[0,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0,0].set_ylim([-0.6,0.6])
    axs[0,1].plot(currency_return_selection["Ether"] , c = 'g')
    axs[0,1].set_title('ETH')
    axs[0,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[0,1].set_ylim([-0.6,0.6])
    axs[1,0].plot(currency_return_selection["Litecoin"] , c = 'y')
    axs[1,0].set_title('LTC')
    axs[1,0].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1,0].set_ylim([-0.6,0.6])
    axs[1,1].plot(currency_return_selection["Dogecoin"] , c = 'b')
    axs[1,1].set_title('DGC')
    axs[1,1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axs[1,1].set_ylim([-0.6,0.6])
    fig.suptitle('Daily Returns Volatility Plots',fontweight ="bold",  fontsize=30)
    st.pyplot(fig)

    st.title('')
    fig , axs = plt.subplots(2 , 2 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
    axs[0,0].hist(currency_return_selection["Bitcoin"] , bins = 100 , color = 'r' , range = (-0.2 , 0.2))
    axs[0,0].set_title('BTC')
    axs[0,1].hist(currency_return_selection["Ether"] , bins = 100 , color = 'g' , range = (-0.2 , 0.2))
    axs[0,1].set_title('ETH')
    axs[1,0].hist(currency_return_selection["Litecoin"] , bins = 100 , color = 'y' , range = (-0.2 , 0.2))
    axs[1,0].set_title('LTC')
    axs[1,1].hist(currency_return_selection["Dogecoin"] , bins = 100 , color = 'b' , range = (-0.2 , 0.2))
    axs[1,1].set_title('DGC')
    fig.suptitle('Daily Returns Histogram Plots',fontweight ="bold" ,  fontsize=30)
    st.pyplot(fig)

    st.title('')
    fig , (ax1, ax2) = plt.subplots(1, 2 , figsize = (16,8) , gridspec_kw = {'hspace' : 0.4 , 'wspace' : 0.1})
    columns = [currency_return_selection["Bitcoin"], currency_return_selection["Ether"], currency_return_selection["Litecoin"], currency_return_selection["Dogecoin"]]
    ax1.set_title('Including Outliers')
    ax1.boxplot(columns , patch_artist=True , meanline = True) 
    ax1.set_xticklabels(["BTC", "ETH", "LTC", "DGC"])
    ax2.set_title('Excluding Outliers')
    ax2.boxplot(columns , patch_artist=True , showfliers=False ,  meanline = True)
    ax2.set_xticklabels(["BTC", "ETH", "LTC", "DGC"])
    fig.suptitle('Daily Returns Box Plots',fontweight ="bold",  fontsize=30)
    st.pyplot(fig)

    st.title('')
    fig,(ax1,ax2) = plt.subplots(1,2, figsize = (16,8) , gridspec_kw = {'hspace' : 0.4 , 'wspace' : 0.3})
    g1 = sns.heatmap(currency_price_selection.corr(),cmap="mako",cbar=False,ax=ax1, annot=True , linewidth=1, linecolor='w' , annot_kws={"size":16 , "weight": "bold"})
    g1.set_ylabel('')
    g1.set_xlabel('')
    g1.set_title('Market Price Correlations')
    g2 = sns.heatmap(currency_return_selection.corr(),cmap="mako",cbar=False,ax=ax2 , annot=True , linewidth=1, linecolor='w',  annot_kws={"size":16 , "weight": "bold"})
    g2.set_ylabel('')
    g2.set_xlabel('')
    g2.set_title('Daily Returns Correlations')
    fig.suptitle('Currency Correlation Plots',fontweight ="bold",  fontsize=30)
    st.pyplot(fig)

    currency_price_selection.index = pd.to_datetime(currency_price_selection.index)
    yearly_lows = currency_price_selection.groupby(currency_price_selection.index.year).min()
    yearly_highs = currency_price_selection.groupby(currency_price_selection.index.year).max()
    
    st.title('')
    st.title('')
    fig, axes = plt.subplots(figsize=(14,8),nrows=2, ncols=2 , gridspec_kw = {'hspace' : 0.4 , 'wspace' : 0.2})
    yearly_highs["Bitcoin"].plot(ax=axes[0,0], kind='bar' )
    for p in axes[0,0].patches:
        axes[0,0].annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axes[0,0].set_ylim([0, 75000])
    axes[0,0].set_xticklabels(["2014", "2015", "2016", "2017" , "2018", "2019", "2020", "2021"])
    yearly_lows["Bitcoin"].plot(ax=axes[0,1], kind='bar' )
    for p in axes[0,1].patches:
        axes[0,1].annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axes[0,1].set_ylim([0, 35000])
    yearly_highs["Ether"].plot(ax=axes[1,0], kind='bar' )
    for p in axes[1,0].patches:
        axes[1,0].annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axes[1,0].set_ylim([0, 5000])
    axes[1,0].set_xticklabels(["2014", "2015", "2016", "2017" , "2018", "2019", "2020", "2021"])
    yearly_lows["Ether"].plot(ax=axes[1,1], kind='bar' )
    for p in axes[1,1].patches:
        axes[1,1].annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axes[1,1].set_ylim([0, 850])
    axes[0,0].set_ylabel('')
    axes[0,0].set_xlabel('')
    axes[0,1].set_ylabel('')
    axes[0,1].set_xlabel('')
    axes[1,0].set_ylabel('')
    axes[1,0].set_xlabel('')
    axes[1,1].set_ylabel('')
    axes[1,1].set_xlabel('')
    axes[0,0].set_title('BTC Yearly All Time Highs')
    axes[0,1].set_title('BTC Yearly All Time Lows')
    axes[1,0].set_title('ETH Yearly All Time Highs')
    axes[1,1].set_title('ETH Yearly All Time Lows')
    fig.suptitle('Yearly All Time Lows & Highs Trends',fontweight ="bold",  fontsize=30)
    st.pyplot(fig)

    st.title("Growth of $1 investment Plots")
    plotly_figure = px.line(data_frame = currency_cum_returns_selection, x = currency_cum_returns_selection.index , y = currency_cum_returns_selection.columns, log_y = False )
    st.plotly_chart(plotly_figure)
    st.title('')



