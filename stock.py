import numpy as np
from datetime import datetime

import time
import os

#For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
#For Stock Data
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
import pandas_datareader
import pandas as pd
from pandas_datareader import data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def getStocks(n):
    #Navigating to the Yahoo stock screener    
    # chrome_driver_path = './chromedriver'
    # driver = webdriver.Chrome(chrome_driver_path)
    # url = 'https://finance.yahoo.com/screener/predefined/aggressive_small_caps?offset=0&count=202'
    # driver.get(url)

    # stock_list = []
    # n += 1
    # for i in range(1, n):
    #     ticker = driver.find_element_by_xpath('//*[@id="scr-res-table"]/div[1]/table/tbody/tr[  {}  ]/td[1]/a'.format(i))
    #     stock_list.append(ticker.text)
    # driver.quit()
    #Using the stock list to predict the future price of the stock a specificed amount of days
    # for i in stock_list:        
    #     predictData(i, 2)
     
    predictData('AAPL',2)
  


def predictData(stock,days,nnn):
    start = datetime(2019, 09, 10)
    end = datetime(2019, 10, 10)
    #Outputting the Historical data into a .csv for later use
    #df = get_historical_data(stock, start,output_format='pandas')
    df = data.get_data_yahoo(stock, start, end)
#     print(stock)
#     print("before",df.head(1))    
    # csv_name = ('Exports/' + stock + '_Export.csv')    
    # df.to_csv(csv_name)

    df['prediction'] = df['Close'].shift(-1)
#     print("after",df.head(1))
#     print(df['prediction'][-2])
    df.dropna(inplace=True)
    forecast_time = int(days)

    X = np.array(df.drop(['prediction'], 1))
    Y = np.array(df['prediction'])
    X = preprocessing.scale(X)
    X_prediction = X[-forecast_time:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
#     print(X_train)
    #Performing the Regression on the training data
    if nnn == 0:
      clf = LinearRegression()
      clf.fit(X_train, Y_train)
      prediction = (clf.predict(X_prediction))
  #     print("Linear Regression")
  #     print("Prediction",prediction)
  #     print("hejfhiodhviodjivd")
      return list(prediction)
    
    
#     print("Dec Tree")
    if nnn == 1 :
      clf = DecisionTreeRegressor()
      clf.fit(X_train, Y_train)
      prediction = (clf.predict(X_prediction))
#     print("Dec Tree")
#     print("Prediction",prediction)
#     print("bcnasdb")
      return list(prediction)

  
#     print("Random Forest")
    if nnn == 2 :
    
      clf = RandomForestRegressor()
      clf.fit(X_train, Y_train)
      prediction = (clf.predict(X_prediction))
#     print("Random Forest")
#     print("Prediction",prediction)
#     print("ashfj")
      return list(prediction)
    
    
#     print("Logistic")
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaler = StandardScaler()
#     X_std = scaler.fit_transform(X_train)
#     clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
#     clf.fit(X_std, Y_train)
#     prediction = (clf.predict(scaler.fit_transform(X_prediction)))
#     print("Logistic")
#     print("Prediction",prediction)
#     print("trgjhkjgt")
#     print(prediction)
    
predictData('AAPL', 5, 1)
#getStocks(5)
