import streamlit as st
import numpy as np
import pandas as pd
from simplified_scrapy import SimplifiedDoc, req
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
    
    features = [
    "Market_Price",
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

    selected_features = col1.multiselect("Select Currency", features, features)
    
    return input_currency , selected_features

def collect_data(input , selected_features):

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

    # Scraping theData for From Below URL & Creating a DataFrame
    html = req.get('https://bitinfocharts.com/comparison/'+input+'-marketcap.html')
    doc = SimplifiedDoc(html)
    js = doc.getElementByText('new Dygraph', tag='script').html
    js = js[js.find('document.getElementById("container"),') + len('document.getElementById("container"),'):]
    js = js[:js.find(', {labels:')] 
    js = js.replace('[new Date("', '').replace('")', '')[1:-2]
    data = [kv.split(',') for kv in js.split('],')]
    Market_Cap = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Transactions = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Amount_Send = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Avg_Transaction_Value = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Avg_Transaction_Fee = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Avg_Block_Time = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Avg_Block_Size = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Miner_Reward = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Mining_Difficulty = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Mining_Profitability = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Hashrate = pd.DataFrame(data)

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
    data = [kv.split(',') for kv in js.split('],')]
    Active_Addresses = pd.DataFrame(data)

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

    feature_selection = Block_Social_Data[selected_features]

    return Block_Social_Data , feature_selection
   
def show_Blockchain_Data_EDA():

    col1 = st.sidebar

    input , selected_features = input_data(col1)
    df , feature_selection = collect_data(input , selected_features)

    feature_selection['Time'] = feature_selection.index

    if st.sidebar.checkbox(label = "Display Blockchain Data"):
        st.title("Blockchain Data")
        st.write(feature_selection)

    st.title("Blockchain Data Plots")
    plotly_figure = px.line(data_frame = feature_selection, x = feature_selection['Time'] , y = selected_features, log_y = True )
    st.plotly_chart(plotly_figure)

    if len(selected_features) > 1:
        cor_feature = col1.selectbox("Select Feature for Correlation Analysis" , selected_features )
        st.title("Blockchain Data Correlation Analysis")
        heatmap = sns.heatmap(feature_selection.corr()[[cor_feature]].sort_values(by=cor_feature, ascending=False), vmin=-1, vmax=1,linewidths = 1, robust = True ,annot=True, cmap='Blues')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    else:
        st.title("")
        st.title("Cannot Display for Single Feature")



    