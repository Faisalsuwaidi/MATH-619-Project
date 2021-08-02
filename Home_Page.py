import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import json
from PIL import Image

def show_home_page():

    st.title("Home Page")
    image = Image.open("Image.jpg")
    st.image(image )
    expander_bar = st.beta_expander("About")
    expander_bar.markdown("""
    * **Student Name:** Fasial Khalid Al-Suwaidi
    * **Student ID:** g201053560
    * **Course:** KFUPM Data Science MX Program Project
    * **Main Libraries Used:** Streamlit, Keras, Statmodels, Scrapy, BeautifulSoup
    * **Main Data Sources:** Bitifochart, Coinmarketcap, Google Trends, Twitter, Yahoo Finance
    """ )
    
    col1 = st.sidebar
    currency_price_unit = 'USD'

    @st.cache
    def load_data():
        cmc = requests.get('https://coinmarketcap.com')
        soup = BeautifulSoup(cmc.content, 'html.parser')
        data = soup.find('script', id='__NEXT_DATA__', type='application/json')
        coins = {}
        coin_data = json.loads(data.contents[0])
        listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']
        for i in listings:
            coins[str(i['id'])] = i['slug']

        coin_name = []
        coin_symbol = []
        market_cap = []
        percent_change_1h = []
        percent_change_24h = []
        percent_change_7d = []
        price = []
        volume_24h = []

        for i in listings:
            coin_name.append(i['slug'])
            coin_symbol.append(i['symbol'])
            price.append(i['quote'][currency_price_unit]['price'])
            percent_change_1h.append(i['quote'][currency_price_unit]['percentChange1h']) 
            percent_change_24h.append(i['quote'][currency_price_unit]['percentChange24h']) 
            percent_change_7d.append(i['quote'][currency_price_unit]['percentChange7d']) 
            market_cap.append(i['quote'][currency_price_unit]['marketCap']) 
            volume_24h.append(i['quote'][currency_price_unit]['volume24h']) 

        df = pd.DataFrame(columns=['coin_name', 'coin_symbol', 'marketCap', 'percentChange1h', 'percentChange24h', 'percentChange7d', 'price', 'volume24h'])
        df['coin_name'] = coin_name
        df['coin_symbol'] = coin_symbol
        df['price'] = price
        df['percentChange1h'] = percent_change_1h
        df['percentChange24h'] = percent_change_24h
        df['percentChange7d'] = percent_change_7d
        df['marketCap'] = market_cap
        df['volume24h'] = volume_24h

        return df

    df = load_data()
    new_df = df[['coin_symbol', 'price', 'marketCap', 'percentChange24h', 'volume24h']]
    new_new_df = new_df[new_df['coin_symbol'].isin(['BTC','ETH','LTC','DOGE'])]

    percent_timeframe = '24h'
    sorted_coin = sorted( new_new_df['coin_symbol'] )
    sort_values = 'Yes'
    selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)

    df_selected_coin = new_new_df[ (new_new_df['coin_symbol'].isin(selected_coin)) ] # Filtering data

    df_coins = df_selected_coin

    st.title('Market Overview for Selected Currencies')

    display = df_coins.set_index(['coin_symbol'])
    st.dataframe(display)

    df_change = pd.concat([df_coins.coin_symbol, df_coins.percentChange24h], axis=1)
    df_change = df_change.set_index('coin_symbol')
    df_change['positive_percent_change_24h'] = df_change['percentChange24h'] > 0

    st.title('Bar Plot of 24 H % Price Change')
    
    if len(selected_coin) > 0:
        if percent_timeframe == '24h':
            if sort_values == 'Yes':
                df_change = df_change.sort_values(by=['percentChange24h'])
            fig, ax = plt.subplots(figsize=(10,5))
            plt.subplots_adjust(top = 1, bottom = 0)
            ax = df_change['percentChange24h'].plot(kind ='bar', color = df_change.positive_percent_change_24h.map({True: 'g', False: 'r'}) ,label = "")
            for p in ax.patches:
                ax.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
            ax.set_ylim([-10, 10])
            ax.set_xlabel('')
            ax.set_ylabel('')
            st.pyplot(fig)
    else:
        st.write("Please Select At Least One Cryptocurrency to Display Results!")


