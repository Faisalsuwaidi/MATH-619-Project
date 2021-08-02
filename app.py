import streamlit as st
from Home_Page import show_home_page
from Market_Price_Return_EDA import show_Market_Price_Return_EDA
from Blockchain_Data_EDA import show_Blockchain_Data_EDA
from Market_Commodity_Data_EDA import show_Market_Commodity_Data_EDA
from Social_Data_EDA import show_Social_Data_EDA
from Time_Series_Model import show_Time_Series_Model

st.sidebar.title("User Inputs")

page = st.sidebar.selectbox("Select Application Page", ("Home Page" , "Market Price & Return EDA" ,"Blockchain Data EDA" , "Market & Commodity Data EDA" , "Social Data EDA" ,"Time Series Model"))

if page == "Time Series Model":
     show_Time_Series_Model()
elif page == "Market Price & Return EDA": 
     show_Market_Price_Return_EDA()
elif page == "Blockchain Data EDA": 
     show_Blockchain_Data_EDA()
elif page == "Market & Commodity Data EDA": 
     show_Market_Commodity_Data_EDA()    
elif page == "Social Data EDA": 
     show_Social_Data_EDA()    
else:
    show_home_page()
    

