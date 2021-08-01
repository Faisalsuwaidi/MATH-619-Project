import streamlit as st
import numpy as np
import pandas as pd
from simplified_scrapy import SimplifiedDoc, req
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from itertools import product
from dateutil.relativedelta import relativedelta
from sklearn.metrics import r2_score

def input_data(col1):

    currencies = (
    "Bitcoin",
    "Ether",
    "Litecoin",
    "Dogecoin",
    )

    models = (
    "SARIMA",
    "ARIMA",
    "AR",
    "MA"
    )

    back_testing = (
    6,
    9,
    12,
    18,
    24
    )

    currency = col1.selectbox("Select Currency", currencies)
    model = col1.selectbox("Select Model", models)
    back_testing_period = col1.selectbox("Select Backtesting Period", back_testing)

    horizon_slider = col1.slider("Investment Horizon (Months)", 1, 24, 6)

    button = st.sidebar.checkbox(label = "Display Model Details")
    
    if currency == "Bitcoin":
        input_currency = "bitcoin"
    elif currency == "Ether": 
        input_currency = "ethereum"
    elif currency == "Litecoin": 
        input_currency = "litecoin"
    else:
        input_currency = "dogecoin"

    return input_currency , horizon_slider , model , back_testing_period , button

def collect_data(input):

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

    # Dropping Any Rows with Missing Values
    Market_Price = Market_Price.dropna()

    # Converting Data from Object to Float
    Market_Price["Market_Price"] = pd.to_numeric(Market_Price["Market_Price"], downcast="float")
    #Market_Price.index = Market_Price.index.strftime('%m/%d/%Y')
    Market_Price = Market_Price['2014-09-30':]

    return Market_Price

def show_Time_Series_Model():

    col1 = st.sidebar
  
    input , horizon_slider , model , back_testing_period , button = input_data(col1)

    df = collect_data(input)

    if model == "AR":

        # Resampling to Daily frequency
        df_daily_plot = df.resample('D').mean()
        df_daily_plot.index = df_daily_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month_plot = df.resample('M').mean()
        df_month_plot.index = df_month_plot.index.strftime('%m/%d/%Y')

        # Resampling to quarterly frequency
        df_Q_plot = df.resample('Q-DEC').mean()
        df_Q_plot.index = df_Q_plot.index.strftime('%m/%d/%Y')
   
        # Resampling to annual frequency
        df_year_plot = df.resample('A-DEC').mean()
        df_year_plot.index = df_year_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month = df.resample('M').mean()

        st.title('')
        fig , axs = plt.subplots(2 , 1 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
        axs[0].plot(df_daily_plot["Market_Price"] , color = 'r' )
        axs[0].set_title('Daily')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[1].plot(df_month_plot["Market_Price"] ,color = 'g' )
        axs[1].set_title('Monthly')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        fig.suptitle('Converting to Monthly Data to Smooth Trends',fontweight ="bold" ,  fontsize=30)
        if button:
            st.pyplot(fig)

        # Seasonal Decomposition
        if button:
            st.title('')
            st.title('Original Sesonal Decomposition')
            seasonal_decompose(df_month.Market_Price).plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p = %f" % adfuller(df_month.Market_Price)[1])

        # Box-Cox Transformations
        df_month['close_box'], lmbda = stats.boxcox(df_month.Market_Price)
        seasonal_decompose(df_month.close_box).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Seasonal differentiation (12 months)
        df_month['box_diff_seasonal_12'] = df_month.close_box - df_month.close_box.shift(12)
        seasonal_decompose(df_month.box_diff_seasonal_12.dropna()).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Regular differentiation
        df_month['box_diff2'] = df_month.box_diff_seasonal_12 - df_month.box_diff_seasonal_12.shift(1)
        if button:
            st.title('')
            st.title('Transformed Sesonal Decomposition')
            seasonal_decompose(df_month.box_diff2[13:]).plot()  
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p=%f" % adfuller(df_month.box_diff2[13:])[1])

        #Autocorrelation and Partial Autocorrelation Plots
        plt.figure(figsize=[20,8])    
        plot_acf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
        plt.figure(figsize=[20,8]) 
        plot_pacf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
    
        # Initial approximation of parameters
        qs = range(0, 3)
        ps = 0
        d = 0
        parameters = product(qs)
        parameters_list = list(parameters)
        len(parameters_list)

        # Model Selection
        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')
        for param in parameters_list:
            try:
                model = SARIMAX(df_month.close_box, order=(ps, d, param[0])).fit(disp=-1)
            except ValueError:
                continue
            aic = model.aic
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        # Best Models
        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        #st.write(result_table.sort_values(by = 'aic', ascending=True).head())
        st.title('Best Model Summary')
        st.title('')
        st.write(best_model.summary())

        # Model Adequacy Check
        best_model.plot_diagnostics(figsize=(15, 12))
        st.title('')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])

        # Inverse Box-Cox Transformation Function
        def invboxcox(y,lmbda):
            if lmbda == 0:
                return(np.exp(y))
            else:
                return(np.exp(np.log(lmbda*y+1)/lmbda))

        btc_month_pred = df_month[['Market_Price']]
        test = pd.date_range(btc_month_pred.index[-1] + relativedelta(months=1), periods = horizon_slider, freq='M')
        future = pd.DataFrame(index=test, columns= df_month.columns)
        btc_month_pred = pd.concat([btc_month_pred, future])
        btc_month_pred['forecast'] = invboxcox(best_model.predict(start=btc_month_pred.index[0], end=future.index[-1]), lmbda)

        fig, ax = plt.subplots(1, figsize=(13, 7))
        plt.plot(btc_month_pred['Market_Price'], color = 'red', label = 'Actual Price');
        plt.plot(btc_month_pred['forecast'], color = 'blue', label = 'Predicted Price');
        ax.legend(loc='best', fontsize=16);
        st.title("")
        st.title("Visualizing Predicitons & Accuracy")
        plt.axvline(x=df_month.index[-1],color='black'  , label= 'Train/Validation Split')
        st.title("")
        st.pyplot(plt)

        # Compute the root mean square error
        y_forecasted = btc_month_pred[df_month.index[0]:df_month.index[-1]].forecast
        y_truth = btc_month_pred[df_month.index[0]:df_month.index[-1]].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))

        # Get the dynamic forecast between dates t1 and t2
        t1 = df_month.index[-1] - relativedelta(months=back_testing_period)
        t2 = df_month.index[-1]
        t0 = t1 - relativedelta(months=36)
        btc_month_dynamic = best_model.get_prediction(start=t1, end=t2, dynamic=True, full_results=True)
        btc_month_pred['dynamic_forecast'] = invboxcox(btc_month_dynamic.predicted_mean, lmbda)

        # Taking 80% confidence interval because the 95% blows out too high to visualise
        pred_dynamic_ci = btc_month_dynamic.conf_int(alpha=0.2)
        pred_dynamic_ci['lower close_box'] = invboxcox(pred_dynamic_ci['lower close_box'], lmbda)
        pred_dynamic_ci['upper close_box'] = invboxcox(pred_dynamic_ci['upper close_box'], lmbda)

        # Plot
        plt.figure(figsize=(15,7))
        btc_month_pred.Market_Price[t0:t2].plot(label='Market_Price')
        btc_month_pred[t1:t2].dynamic_forecast.plot(color='r', ls='--', label='Predicted Price')
        plt.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        plt.fill_betweenx(plt.ylim(), t1, t2, alpha=.1, zorder=-1)
        plt.legend()
        st.title("")
        st.title("Visualizing Backtesting Impact")
        st.title("")
        plt.ylabel('USD')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # Compute the root mean square error for back testing
        y_forecasted = btc_month_pred[t1:t2].dynamic_forecast
        y_truth = btc_month_pred[t1:t2].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))
    
    elif model == "MA":

        # Resampling to Daily frequency
        df_daily_plot = df.resample('D').mean()
        df_daily_plot.index = df_daily_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month_plot = df.resample('M').mean()
        df_month_plot.index = df_month_plot.index.strftime('%m/%d/%Y')

        # Resampling to quarterly frequency
        df_Q_plot = df.resample('Q-DEC').mean()
        df_Q_plot.index = df_Q_plot.index.strftime('%m/%d/%Y')
   
        # Resampling to annual frequency
        df_year_plot = df.resample('A-DEC').mean()
        df_year_plot.index = df_year_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month = df.resample('M').mean()

        st.title('')
        fig , axs = plt.subplots(2 , 1 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
        axs[0].plot(df_daily_plot["Market_Price"] , color = 'r' )
        axs[0].set_title('Daily')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[1].plot(df_month_plot["Market_Price"] ,color = 'g' )
        axs[1].set_title('Monthly')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        fig.suptitle('Converting to Monthly Data to Smooth Trends',fontweight ="bold" ,  fontsize=30)
        if button:
            st.pyplot(fig)

        # Seasonal Decomposition
        if button:
            st.title('')
            st.title('Original Sesonal Decomposition')
            seasonal_decompose(df_month.Market_Price).plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p = %f" % adfuller(df_month.Market_Price)[1])

        # Box-Cox Transformations
        df_month['close_box'], lmbda = stats.boxcox(df_month.Market_Price)
        seasonal_decompose(df_month.close_box).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Seasonal differentiation (12 months)
        df_month['box_diff_seasonal_12'] = df_month.close_box - df_month.close_box.shift(12)
        seasonal_decompose(df_month.box_diff_seasonal_12.dropna()).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Regular differentiation
        df_month['box_diff2'] = df_month.box_diff_seasonal_12 - df_month.box_diff_seasonal_12.shift(1)
        if button:
            st.title('')
            st.title('Transformed Sesonal Decomposition')
            seasonal_decompose(df_month.box_diff2[13:]).plot()  
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p=%f" % adfuller(df_month.box_diff2[13:])[1])

        #Autocorrelation and Partial Autocorrelation Plots
        plt.figure(figsize=[20,8])    
        plot_acf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
        plt.figure(figsize=[20,8]) 
        plot_pacf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
    
        # Initial approximation of parameters
        qs = 0
        ps = range(0, 3)
        d = 0
        parameters = product(ps)
        parameters_list = list(parameters)
        len(parameters_list)

        # Model Selection
        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')
        for param in parameters_list:
            try:
                model = SARIMAX(df_month.close_box, order=(param[0], d, qs)).fit(disp=-1)
            except ValueError:
                continue
            aic = model.aic
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        # Best Models
        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        #st.write(result_table.sort_values(by = 'aic', ascending=True).head())
        st.title('Best Model Summary')
        st.title('')
        st.write(best_model.summary())

        # Model Adequacy Check
        best_model.plot_diagnostics(figsize=(15, 12))
        st.title('')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])

        # Inverse Box-Cox Transformation Function
        def invboxcox(y,lmbda):
            if lmbda == 0:
                return(np.exp(y))
            else:
                return(np.exp(np.log(lmbda*y+1)/lmbda))

        btc_month_pred = df_month[['Market_Price']]
        test = pd.date_range(btc_month_pred.index[-1] + relativedelta(months=1), periods = horizon_slider, freq='M')
        future = pd.DataFrame(index=test, columns= df_month.columns)
        btc_month_pred = pd.concat([btc_month_pred, future])
        btc_month_pred['forecast'] = invboxcox(best_model.predict(start=btc_month_pred.index[0], end=future.index[-1]), lmbda)

        fig, ax = plt.subplots(1, figsize=(13, 7))
        plt.plot(btc_month_pred['Market_Price'], color = 'red', label = 'Actual Price');
        plt.plot(btc_month_pred['forecast'], color = 'blue', label = 'Predicted Price');
        ax.legend(loc='best', fontsize=16);
        st.title("")
        st.title("Visualizing Predicitons & Accuracy")
        plt.axvline(x=df_month.index[-1],color='black'  , label= 'Train/Validation Split')
        st.title("")
        st.pyplot(plt)

        # Compute the root mean square error
        y_forecasted = btc_month_pred[df_month.index[0]:df_month.index[-1]].forecast
        y_truth = btc_month_pred[df_month.index[0]:df_month.index[-1]].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))


        # Get the dynamic forecast between dates t1 and t2
        t1 = df_month.index[-1] - relativedelta(months=back_testing_period)
        t2 = df_month.index[-1]
        t0 = t1 - relativedelta(months=36)
        btc_month_dynamic = best_model.get_prediction(start=t1, end=t2, dynamic=True, full_results=True)
        btc_month_pred['dynamic_forecast'] = invboxcox(btc_month_dynamic.predicted_mean, lmbda)

        # Taking 80% confidence interval because the 95% blows out too high to visualise
        pred_dynamic_ci = btc_month_dynamic.conf_int(alpha=0.2)
        pred_dynamic_ci['lower close_box'] = invboxcox(pred_dynamic_ci['lower close_box'], lmbda)
        pred_dynamic_ci['upper close_box'] = invboxcox(pred_dynamic_ci['upper close_box'], lmbda)

        # Plot
        plt.figure(figsize=(15,7))
        btc_month_pred.Market_Price[t0:t2].plot(label='Market_Price')
        btc_month_pred[t1:t2].dynamic_forecast.plot(color='r', ls='--', label='Predicted Price')
        plt.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        plt.fill_betweenx(plt.ylim(), t1, t2, alpha=.1, zorder=-1)
        plt.legend()
        st.title("")
        st.title("Visualizing Backtesting Impact")
        st.title("")
        plt.ylabel('USD')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # Compute the root mean square error for back testing
        y_forecasted = btc_month_pred[t1:t2].dynamic_forecast
        y_truth = btc_month_pred[t1:t2].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))
    
    elif model == "ARIMA":

        # Resampling to Daily frequency
        df_daily_plot = df.resample('D').mean()
        df_daily_plot.index = df_daily_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month_plot = df.resample('M').mean()
        df_month_plot.index = df_month_plot.index.strftime('%m/%d/%Y')

        # Resampling to quarterly frequency
        df_Q_plot = df.resample('Q-DEC').mean()
        df_Q_plot.index = df_Q_plot.index.strftime('%m/%d/%Y')
   
        # Resampling to annual frequency
        df_year_plot = df.resample('A-DEC').mean()
        df_year_plot.index = df_year_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month = df.resample('M').mean()

        st.title('')
        fig , axs = plt.subplots(2 , 1 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
        axs[0].plot(df_daily_plot["Market_Price"] , color = 'r' )
        axs[0].set_title('Daily')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[1].plot(df_month_plot["Market_Price"] ,color = 'g' )
        axs[1].set_title('Monthly')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        fig.suptitle('Converting to Monthly Data to Smooth Trends',fontweight ="bold" ,  fontsize=30)
        if button:
            st.pyplot(fig)

        # Seasonal Decomposition
        if button:
            st.title('')
            st.title('Original Sesonal Decomposition')
            seasonal_decompose(df_month.Market_Price).plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p = %f" % adfuller(df_month.Market_Price)[1])

        # Box-Cox Transformations
        df_month['close_box'], lmbda = stats.boxcox(df_month.Market_Price)
        seasonal_decompose(df_month.close_box).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Seasonal differentiation (12 months)
        df_month['box_diff_seasonal_12'] = df_month.close_box - df_month.close_box.shift(12)
        seasonal_decompose(df_month.box_diff_seasonal_12.dropna()).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Regular differentiation
        df_month['box_diff2'] = df_month.box_diff_seasonal_12 - df_month.box_diff_seasonal_12.shift(1)
        if button:
            st.title('')
            st.title('Transformed Sesonal Decomposition')
            seasonal_decompose(df_month.box_diff2[13:]).plot()  
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p=%f" % adfuller(df_month.box_diff2[13:])[1])

        #Autocorrelation and Partial Autocorrelation Plots
        plt.figure(figsize=[20,8])    
        plot_acf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
        plt.figure(figsize=[20,8]) 
        plot_pacf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
    
        # Initial approximation of parameters
        qs = range(0, 3)
        ps = range(0, 3)
        d = 1
        parameters = product(ps, qs)
        parameters_list = list(parameters)
        len(parameters_list)

        # Model Selection
        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')
        for param in parameters_list:
            try:
                model = SARIMAX(df_month.close_box, order=(param[0], d, param[1])).fit(disp=-1)
            except ValueError:
                continue
            aic = model.aic
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        # Best Models
        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        #st.write(result_table.sort_values(by = 'aic', ascending=True).head())
        st.title('Best Model Summary')
        st.title('')
        st.write(best_model.summary())

        # Model Adequacy Check
        best_model.plot_diagnostics(figsize=(15, 12))
        st.title('')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])

        # Inverse Box-Cox Transformation Function
        def invboxcox(y,lmbda):
            if lmbda == 0:
                return(np.exp(y))
            else:
                return(np.exp(np.log(lmbda*y+1)/lmbda))

        btc_month_pred = df_month[['Market_Price']]
        test = pd.date_range(btc_month_pred.index[-1] + relativedelta(months=1), periods = horizon_slider, freq='M')
        future = pd.DataFrame(index=test, columns= df_month.columns)
        btc_month_pred = pd.concat([btc_month_pred, future])
        btc_month_pred['forecast'] = invboxcox(best_model.predict(start=btc_month_pred.index[0], end=future.index[-1]), lmbda)

        fig, ax = plt.subplots(1, figsize=(13, 7))
        plt.plot(btc_month_pred['Market_Price'], color = 'red', label = 'Actual Price');
        plt.plot(btc_month_pred['forecast'], color = 'blue', label = 'Predicted Price');
        ax.legend(loc='best', fontsize=16);
        st.title("")
        st.title("Visualizing Predicitons & Accuracy")
        plt.axvline(x=df_month.index[-1],color='black'  , label= 'Train/Validation Split')
        st.title("")
        st.pyplot(plt)

        # Compute the root mean square error
        y_forecasted = btc_month_pred[df_month.index[0]:df_month.index[-1]].forecast
        y_truth = btc_month_pred[df_month.index[0]:df_month.index[-1]].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))

        # Get the dynamic forecast between dates t1 and t2
        t1 = df_month.index[-1] - relativedelta(months=back_testing_period)
        t2 = df_month.index[-1]
        t0 = t1 - relativedelta(months=36)
        btc_month_dynamic = best_model.get_prediction(start=t1, end=t2, dynamic=True, full_results=True)
        btc_month_pred['dynamic_forecast'] = invboxcox(btc_month_dynamic.predicted_mean, lmbda)

        # Taking 80% confidence interval because the 95% blows out too high to visualise
        pred_dynamic_ci = btc_month_dynamic.conf_int(alpha=0.2)
        pred_dynamic_ci['lower close_box'] = invboxcox(pred_dynamic_ci['lower close_box'], lmbda)
        pred_dynamic_ci['upper close_box'] = invboxcox(pred_dynamic_ci['upper close_box'], lmbda)

        # Plot
        plt.figure(figsize=(15,7))
        btc_month_pred.Market_Price[t0:t2].plot(label='Market_Price')
        btc_month_pred[t1:t2].dynamic_forecast.plot(color='r', ls='--', label='Predicted Price')
        plt.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        plt.fill_betweenx(plt.ylim(), t1, t2, alpha=.1, zorder=-1)
        plt.legend()
        st.title("")
        st.title("Visualizing Backtesting Impact")
        st.title("")
        plt.ylabel('USD')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # Compute the root mean square error for back testing
        y_forecasted = btc_month_pred[t1:t2].dynamic_forecast
        y_truth = btc_month_pred[t1:t2].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))

    elif model == "SARIMA":  

        # Resampling to Daily frequency
        df_daily_plot = df.resample('D').mean()
        df_daily_plot.index = df_daily_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month_plot = df.resample('M').mean()
        df_month_plot.index = df_month_plot.index.strftime('%m/%d/%Y')

        # Resampling to quarterly frequency
        df_Q_plot = df.resample('Q-DEC').mean()
        df_Q_plot.index = df_Q_plot.index.strftime('%m/%d/%Y')
   
        # Resampling to annual frequency
        df_year_plot = df.resample('A-DEC').mean()
        df_year_plot.index = df_year_plot.index.strftime('%m/%d/%Y')

        # Resampling to monthly frequency
        df_month = df.resample('M').mean()

        st.title('')
        fig , axs = plt.subplots(2 , 1 , figsize = (14,8) , gridspec_kw = {'hspace' : 0.2 , 'wspace' : 0.1})
        axs[0].plot(df_daily_plot["Market_Price"] , color = 'r' )
        axs[0].set_title('Daily')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        axs[1].plot(df_month_plot["Market_Price"] ,color = 'g' )
        axs[1].set_title('Monthly')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        fig.suptitle('Converting to Monthly Data to Smooth Trends',fontweight ="bold" ,  fontsize=30)
        if button:
            st.pyplot(fig)

        # Seasonal Decomposition
        if button:
            st.title('')
            st.title('Original Sesonal Decomposition')
            seasonal_decompose(df_month.Market_Price).plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p = %f" % adfuller(df_month.Market_Price)[1])

        # Box-Cox Transformations
        df_month['close_box'], lmbda = stats.boxcox(df_month.Market_Price)
        seasonal_decompose(df_month.close_box).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Seasonal differentiation (12 months)
        df_month['box_diff_seasonal_12'] = df_month.close_box - df_month.close_box.shift(12)
        seasonal_decompose(df_month.box_diff_seasonal_12.dropna()).plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Regular differentiation
        df_month['box_diff2'] = df_month.box_diff_seasonal_12 - df_month.box_diff_seasonal_12.shift(1)
        if button:
            st.title('')
            st.title('Transformed Sesonal Decomposition')
            seasonal_decompose(df_month.box_diff2[13:]).plot()  
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write("Dickey–Fuller test: p=%f" % adfuller(df_month.box_diff2[13:])[1])

        #Autocorrelation and Partial Autocorrelation Plots
        plt.figure(figsize=[20,8])    
        plot_acf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()
        plt.figure(figsize=[20,8]) 
        plot_pacf(df_month.box_diff2[13:].values.squeeze())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if button:
            st.pyplot()

        # Initial approximation of parameters
        Qs = range(0, 2)
        qs = range(0, 3)
        Ps = range(0, 3)
        ps = range(0, 3)
        D = 1
        d = 1
        parameters = product(ps, qs, Ps, Qs)
        parameters_list = list(parameters)
        len(parameters_list)

        # Model Selection
        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')
        for param in parameters_list:
            try:
                model = SARIMAX(df_month.close_box, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
            except ValueError:
                continue
            aic = model.aic
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])
    
        # Best Models
        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        #st.write(result_table.sort_values(by = 'aic', ascending=True).head())
        st.title('Best Model Summary')
        st.title('')
        st.write(best_model.summary())

        # Model Adequacy Check
        best_model.plot_diagnostics(figsize=(15, 12))
        st.title('')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.write("Dickey–Fuller test:: p=%f" % adfuller(best_model.resid[13:])[1])

        # Inverse Box-Cox Transformation Function
        def invboxcox(y,lmbda):
            if lmbda == 0:
                return(np.exp(y))
            else:
                return(np.exp(np.log(lmbda*y+1)/lmbda))

        btc_month_pred = df_month[['Market_Price']]
        test = pd.date_range(btc_month_pred.index[-1] + relativedelta(months=1), periods = horizon_slider, freq='M')
        future = pd.DataFrame(index=test, columns= df_month.columns)
        btc_month_pred = pd.concat([btc_month_pred, future])
        btc_month_pred['forecast'] = invboxcox(best_model.predict(start=btc_month_pred.index[0], end=future.index[-1]), lmbda)

        fig, ax = plt.subplots(1, figsize=(13, 7))
        plt.plot(btc_month_pred['Market_Price'], color = 'red', label = 'Actual Price');
        plt.plot(btc_month_pred['forecast'], color = 'blue', label = 'Predicted Price');
        ax.legend(loc='best', fontsize=16);
        st.title("")
        st.title("Visualizing Predicitons & Accuracy")
        plt.axvline(x=df_month.index[-1],color='black'  , label= 'Train/Validation Split')
        st.title("")
        st.pyplot(plt)

        # Compute the root mean square error for model
        y_forecasted = btc_month_pred[df_month.index[0]:df_month.index[-1]].forecast
        y_truth = btc_month_pred[df_month.index[0]:df_month.index[-1]].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))
   

        # Get the dynamic forecast between dates t1 and t2
        t1 = df_month.index[-1] - relativedelta(months=back_testing_period)
        t2 = df_month.index[-1]
        t0 = t1 - relativedelta(months=12)
        btc_month_dynamic = best_model.get_prediction(start=t1, end=t2, dynamic=True, full_results=True)
        btc_month_pred['dynamic_forecast'] = invboxcox(btc_month_dynamic.predicted_mean, lmbda)

        # Taking 80% confidence interval because the 95% blows out too high to visualise
        pred_dynamic_ci = btc_month_dynamic.conf_int(alpha=0.2)
        pred_dynamic_ci['lower close_box'] = invboxcox(pred_dynamic_ci['lower close_box'], lmbda)
        pred_dynamic_ci['upper close_box'] = invboxcox(pred_dynamic_ci['upper close_box'], lmbda)

        # Plot
        plt.figure(figsize=(15,7))
        btc_month_pred.Market_Price[t0:t2].plot(label='Market_Price')
        btc_month_pred[t1:t2].dynamic_forecast.plot(color='r', ls='--', label='Predicted Price')
        plt.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
        plt.fill_betweenx(plt.ylim(), t1, t2, alpha=.1, zorder=-1)
        plt.legend()
        st.title("")
        st.title("Visualizing Backtesting Impact")
        st.title("")
        plt.ylabel('USD')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # Compute the root mean square error for back testing
        y_forecasted = btc_month_pred[t1:t2].dynamic_forecast
        y_truth = btc_month_pred[t1:t2].Market_Price
        rmse = np.sqrt(((y_forecasted - y_truth) ** 2).mean())
        st.write('R Squared Score: {}'.format(round(r2_score(y_forecasted, y_truth), 2)))

    
 
