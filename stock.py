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
import random
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from newsapi.newsapi_client import NewsApiClient

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import time
import keras
from keras.models import load_model
from keras.models import model_from_json
import json
from wordcloud import WordCloud
from PIL import Image
import base64
import shutil

def predict_news(X_test):
   


    json_file = open('advance.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("advance.h5")
    loaded_model._make_predict_function()
    print("Loaded model from disk")

    prediction = loaded_model.predict(X_test)
    return prediction




def getWordCloud(q1):

 
    text = "Insert Your text here"
    try:
        newsapi = NewsApiClient(api_key='Your Key')
    except:
        newsapi = NewsApiClient(api_key='Your another Key incase of exception')
        
        
    end = date.today()
    start =  date.today() - relativedelta(days=+10)

    s1=str(start).split(" ")[0]
    e1=str(end).split(" ")[0]

    

    try:
        news = newsapi.get_everything(q=q1,from_param=s1,
                                              to=e1,language='en',sort_by='popularity',page_size=100,page=1)

        #arranging the news articles in a numpy array
        news_data = [] #np.array(['publishedAt','title','description','content','url'])
        index = 0
        for i in news['articles']:
          k=0
          data = []
          while k!=1:
            data.append(i["publishedAt"])
            data.append(i["title"])
            data.append(i["description"])
            data.append(i["content"])
            data.append(i["url"])
            data.append(i["urlToImage"])
            k=1
          news_data.append(data)
          
        nd = np.array(news_data)


        data = pd.DataFrame(nd,columns=['Date','Title','Description','Content','URL','Image'])


        #Spliting the date and time of each data field.
        new = data["Date"].str.split("T", n = 1, expand = True)
        new1 = new[1].str.split("Z", n=1,expand=True)
        data = data.drop("Date",axis=1)
        data.insert(loc=0, column='Date', value=new[0])
        data.insert(loc=1, column='Time', value=new1[0])


        data['Date'] =pd.to_datetime(data.Date)
        dff=data[['Date','Description']].groupby('Date').sum()

        text=" ".join(list(dff['Description']))+q1*15
        
        wave_mask = np.array(Image.open( "assets/cloud.jpg"))
        wordcloud = WordCloud(mask=wave_mask, width=512, height=512, colormap="Greens").generate(text)
        in1=str(random.randint(0,100000000000000))
        in2=str(random.randint(0,100000000000000))
        in3=str(random.randint(0,100000000000000))
        in4=in1+in2+in3
              
        shutil.rmtree('assets/images')
        os.mkdir("assets/images")
        # Try
        wordcloud.to_file("assets/images/"+in4+".jpg")

        print("saved...")


        
##        img1 = base64.b64encode(open("assets/images/"+in4+".jpg", 'rb').read())
        with open("assets/images/"+in4+".jpg", "rb") as imageFile:
            img1=base64.b64encode(imageFile.read()).decode("utf-8")
            
        print("Yes\n\n")
        return [img1]
        
        

    except Exception as e:
        print(e)
        
        try:
            newsapi = NewsApiClient(api_key='Your another key')
            news = newsapi.get_everything(q=q1,from_param=s1,
                                              to=e1,language='en',sort_by='popularity',page_size=100,page=1)
            print(news,"1")

            #arranging the news articles in a numpy array
            news_data = [] #np.array(['publishedAt','title','description','content','url'])
            index = 0
            for i in news['articles']:
              k=0
              data = []
              while k!=1:
                data.append(i["publishedAt"])
                data.append(i["title"])
                data.append(i["description"])
                data.append(i["content"])
                data.append(i["url"])
                data.append(i["urlToImage"])
                k=1
              news_data.append(data)
              
            nd = np.array(news_data)


            data = pd.DataFrame(nd,columns=['Date','Title','Description','Content','URL','Image'])


            #Spliting the date and time of each data field.
            new = data["Date"].str.split("T", n = 1, expand = True)
            new1 = new[1].str.split("Z", n=1,expand=True)
            data = data.drop("Date",axis=1)
            data.insert(loc=0, column='Date', value=new[0])
            data.insert(loc=1, column='Time', value=new1[0])


            data['Date'] =pd.to_datetime(data.Date)
    ##        data = data.sort_values(by='Date')

            data=data.sort_values(by=['Date'],ascending=False)

            return data
        except:
            return []
        return []




def getTrendingNews(q1):
    
    try:
        newsapi = NewsApiClient(api_key='Your Key')
    except:
        newsapi = NewsApiClient(api_key='Your another key incase of exception')
    end = date.today()
    start =  date.today() - relativedelta(days=+10)

    s1=str(start).split(" ")[0]
    e1=str(end).split(" ")[0]

    

    try:
        news = newsapi.get_everything(q=q1,from_param=s1,
                                              to=e1,language='en',sort_by='popularity',page_size=100,page=1)
        print(news,"1")

        #arranging the news articles in a numpy array
        news_data = [] #np.array(['publishedAt','title','description','content','url'])
        index = 0
        for i in news['articles']:
          k=0
          data = []
          while k!=1:
            data.append(i["publishedAt"])
            data.append(i["title"])
            data.append(i["description"])
            data.append(i["content"])
            data.append(i["url"])
            data.append(i["urlToImage"])
            k=1
          news_data.append(data)
          
        nd = np.array(news_data)


        data = pd.DataFrame(nd,columns=['Date','Title','Description','Content','URL','Image'])


        #Spliting the date and time of each data field.
        new = data["Date"].str.split("T", n = 1, expand = True)
        new1 = new[1].str.split("Z", n=1,expand=True)
        data = data.drop("Date",axis=1)
        data.insert(loc=0, column='Date', value=new[0])
        data.insert(loc=1, column='Time', value=new1[0])


        data['Date'] =pd.to_datetime(data.Date)
##        data = data.sort_values(by='Date')

        data=data.sort_values(by=['Date'],ascending=False)

        return data
    except Exception as e:
        print(e)


        try:
            newsapi = NewsApiClient(api_key='Your another Key')
            news = newsapi.get_everything(q=q1,from_param=s1,
                                              to=e1,language='en',sort_by='popularity',page_size=100,page=1)
            print(news,"1")

            #arranging the news articles in a numpy array
            news_data = [] #np.array(['publishedAt','title','description','content','url'])
            index = 0
            for i in news['articles']:
              k=0
              data = []
              while k!=1:
                data.append(i["publishedAt"])
                data.append(i["title"])
                data.append(i["description"])
                data.append(i["content"])
                data.append(i["url"])
                data.append(i["urlToImage"])
                k=1
              news_data.append(data)
              
            nd = np.array(news_data)


            data = pd.DataFrame(nd,columns=['Date','Title','Description','Content','URL','Image'])


            #Spliting the date and time of each data field.
            new = data["Date"].str.split("T", n = 1, expand = True)
            new1 = new[1].str.split("Z", n=1,expand=True)
            data = data.drop("Date",axis=1)
            data.insert(loc=0, column='Date', value=new[0])
            data.insert(loc=1, column='Time', value=new1[0])


            data['Date'] =pd.to_datetime(data.Date)
    ##        data = data.sort_values(by='Date')

            data=data.sort_values(by=['Date'],ascending=False)

            return data
        except:
            return []
        return []





def createVector(lTemp):
    combined = np.array(lTemp.loc[:,['Title']])
    z=combined.shape[0]
    corpus = combined.reshape(z).tolist()
    corpus

    tfidfVec = TfidfVectorizer(strip_accents = 'ascii', stop_words = "english", analyzer='word', min_df = 0.005, sublinear_tf = False)
    freq_Vec = tfidfVec.fit_transform(corpus)

    return freq_Vec


def getNews(df,s1,e1,q1):


    try:
        newsapi = NewsApiClient(api_key='Your key')
    except:
        newsapi = NewsApiClient(api_key='Your another key incase of exception')
    
    #newsapi = NewsApiClient(api_key='Your key')

    try:
        news = newsapi.get_everything(q=q1,from_param=s1,
                                              to=e1,language='en',sort_by='popularity',page_size=100,page=1)

        #arranging the news articles in a numpy array
        news_data = [] #np.array(['publishedAt','title','description','content','url'])
        index = 0
        for i in news['articles']:
          k=0
          data = []
          while k!=1:
            data.append(i["publishedAt"])
            data.append(i["title"])
            data.append(i["description"])
            data.append(i["content"])
            data.append(i["url"])
            k=1
          news_data.append(data)
          
        nd = np.array(news_data)


        data = pd.DataFrame(nd,columns=['Date','Title','Description','Content','URL'])


        #Spliting the date and time of each data field.
        new = data["Date"].str.split("T", n = 1, expand = True)
        new1 = new[1].str.split("Z", n=1,expand=True)
        data = data.drop("Date",axis=1)
        data.insert(loc=0, column='Date', value=new[0])
        data.insert(loc=1, column='Time', value=new1[0])


        data['Date'] =pd.to_datetime(data.Date)
        data = data.sort_values(by='Date')
        dff=data[['Date','Title']].groupby('Date').sum()

 
        
        lTemp={'Date':[],'Title':[]}

        for i in range(len(df)):
            lTemp['Date'].append(df.index[i])
            lTemp['Title'].append(" ")
            
              


        for i in range(len(lTemp['Date'])):
            if lTemp['Date'][i] in list(dff.index):
                lTemp['Title'][i]=dff['Title'][list(dff.index).index(lTemp['Date'][i])]
            else:
                pass
            
        lTemp=pd.DataFrame(lTemp)

        print(lTemp.head())
        
        freq_Vec=createVector(lTemp)

        if freq_Vec.shape[1]>657:
            freq_Vec=freq_Vec.tocsr()[:,0:658]
        else:
            q=np.array(freq_Vec.todense())
            b=np.zeros((freq_Vec.shape[0],657-freq_Vec.shape[1]))
            p = np.concatenate((q,b),axis=1)
            freq_Vec = csr_matrix(p)
            
        X_test = freq_Vec.toarray()
        mean = np.mean(X_test)
        X_test -= mean


        maxScore=max(list(df['Volume']))
        minScore=min(list(df['Volume']))
        finScore=(maxScore+minScore)//2     
        prediction=predict_news(X_test)
        ypred=prediction
        pol_score = []
        for i in range(len(ypred)):
            pol_score.append((ypred[i,1]-ypred[i,0])*finScore)

        df['Polarity']=pol_score

        return df
    except Exception as e:
        print(e)
        df['Polarity']=[0 for i in range(len(df))]
        

        try:
            newsapi = NewsApiClient(api_key='Another key incase of exception')
            news = newsapi.get_everything(q=q1,from_param=s1,
                                              to=e1,language='en',sort_by='popularity',page_size=100,page=1)
            print(news,"1")

            #arranging the news articles in a numpy array
            news_data = [] #np.array(['publishedAt','title','description','content','url'])
            index = 0
            for i in news['articles']:
              k=0
              data = []
              while k!=1:
                data.append(i["publishedAt"])
                data.append(i["title"])
                data.append(i["description"])
                data.append(i["content"])
                data.append(i["url"])
                data.append(i["urlToImage"])
                k=1
              news_data.append(data)
              
            nd = np.array(news_data)


            data = pd.DataFrame(nd,columns=['Date','Title','Description','Content','URL','Image'])


            #Spliting the date and time of each data field.
            new = data["Date"].str.split("T", n = 1, expand = True)
            new1 = new[1].str.split("Z", n=1,expand=True)
            data = data.drop("Date",axis=1)
            data.insert(loc=0, column='Date', value=new[0])
            data.insert(loc=1, column='Time', value=new1[0])


            data['Date'] =pd.to_datetime(data.Date)
    ##        data = data.sort_values(by='Date')

            data=data.sort_values(by=['Date'])
            dff=data[['Date','Title']].groupby('Date').sum()

 
            
            lTemp={'Date':[],'Title':[]}

            for i in range(len(df)):
                lTemp['Date'].append(df.index[i])
                lTemp['Title'].append(" ")
                
                  


            for i in range(len(lTemp['Date'])):
                if lTemp['Date'][i] in list(dff.index):
                    lTemp['Title'][i]=dff['Title'][list(dff.index).index(lTemp['Date'][i])]
                else:
                    pass
                
            lTemp=pd.DataFrame(lTemp)

            print(lTemp.head())
            
            freq_Vec=createVector(lTemp)

            if freq_Vec.shape[1]>657:
                freq_Vec=freq_Vec.tocsr()[:,0:658]
            else:
                q=np.array(freq_Vec.todense())
                b=np.zeros((freq_Vec.shape[0],657-freq_Vec.shape[1]))
                p = np.concatenate((q,b),axis=1)
                freq_Vec = csr_matrix(p)
                
            X_test = freq_Vec.toarray()
            mean = np.mean(X_test)
            X_test -= mean


            maxScore=max(list(df['Volume']))
            minScore=min(list(df['Volume']))
            finScore=(maxScore+minScore)//2     
            prediction=predict_news(X_test)
            ypred=prediction
            pol_score = []
            for i in range(len(ypred)):
                pol_score.append((ypred[i,1]-ypred[i,0])*finScore)

            df['Polarity']=pol_score

        
            return df

        except:
            df['Polarity']=[0 for i in range(len(df))]
            return df
        return []


  


def predictData(stock,days,nnn,labelsStock):
##    end = date.today()
##    #start = datetime(2019, 09, 10)
##    start =  date.today() - relativedelta(days=+15)
    #Outputting the Historical data into a .csv for later use
    #df = get_historical_data(stock, start,output_format='pandas')
    end = date.today()
    start =  date.today() - relativedelta(days=+14)
    start1=date.today() - relativedelta(days=+15)
    flagsD=0
    cntD=0
    cntDE=0
    cntDS=0
    while(flagsD==0):
        try:
            df = data.get_data_yahoo(stock, start, end)
            flagsD=1
            break
        except:
            if 14>cntD>=7:
                cntDE+=1
                end = date.today()- relativedelta(days=+cntDE)
                start =  date.today() - relativedelta(days=+14)
                start1=date.today() - relativedelta(days=+15)
                if cntD==13:
                    cntDE=0
                    cntDS=0
            elif cntDE==0 and cntD<7:
                cntDS+=1
                start =  date.today() - relativedelta(days=+14+cntDS)
                start1=date.today() - relativedelta(days=+15+cntDS)

            else:
                cntDE+=1
                end = date.today()- relativedelta(days=+cntDE)
                cntDS+=1
                start =  date.today() - relativedelta(days=+14+cntDS)
                start1=date.today() - relativedelta(days=+15+cntDS)
            cntD+=1
                
                
            
            
        
    print(df.head())
    s1=df.index[0]
    e1=df.index[-1]

    s1=str(s1).split(" ")[0]
    e1=str(e1).split(" ")[0]
    
##    s1=str(s1).split(" ")[0]
##    e1=str(e1).split(" ")[0]

    q1=str(labelsStock)#+" "+str(stock).split(".")[0]+" India"
    print(q1)
    df=getNews(df,s1,e1,q1)

#     print(stock)
#     print("before",df.head(1))    
    # csv_name = ('Exports/' + stock + '_Export.csv')    
    # df.to_csv(csv_name)

    print("---------------")
    print(df)
    print("---------------")
    
    df['prediction'] = df['Close'].shift(-1)
#     print("after",df.head(1))
#     print(df['prediction'][-2])
    print(df.head())
    df.dropna(inplace=True)
    print(df.head())
    forecast_time = int(1)

    X = np.array(df.drop(['prediction'], 1))
    print(X)
    print(len(X))
    Y = np.array(df['prediction'])
    print(Y)
    print(len(Y))
    X = preprocessing.scale(X)
    X=  preprocessing.normalize(X)
    print(X)
    X_prediction = X[-forecast_time:]
    X_train, Y_train=X[:-1],Y[:-1]
        
    print("-----------------")
    print(df)
    print("-----------------")
    


    
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.00000000000000000000000001,shuffle=False)
#     print(X_train)
    #Performing the Regression on the training data
    if nnn == 0:
        clf = LinearRegression()
        clf.fit(X_train, Y_train)
        prediction = (clf.predict(X_prediction))
        
        l=[prediction[0]]
        lsr=list(df['prediction'])
        lsr.pop(0)
        lsr.append(prediction[0])
        for i in range(2,6):
            
            df['prediction'] = lsr

            
        #df.dropna(inplace=True)

            print("------------------")
            print(df)
            print("i--->",i)
            print("------------------")
            X = np.array(df.drop(['prediction'], 1))
            print(X)
            print(len(X))
            Y = np.array(df['prediction'])
            print(Y)
            print(len(Y))
            X = preprocessing.scale(X)
            X=  preprocessing.normalize(X)
            print(X)
            X_prediction = X[-forecast_time:]
            X_train, Y_train=X[:-1],Y[:-1]
            clf = DecisionTreeRegressor()
            clf.fit(X_train, Y_train)
            prediction = (clf.predict(X_prediction))
            lsr.pop(0)
            lsr.append(prediction[0])
            l.append(prediction[0])

  #     print("Linear Regression")
  #     print("Prediction",prediction)
  #     print("hejfhiodhviodjivd")
        return list(l)
    
    
#     print("Dec Tree")
    if nnn == 1 :
        clf = DecisionTreeRegressor()
        clf.fit(X_train, Y_train)
        prediction = (clf.predict(X_prediction))

        l=[prediction[0]]
        lsr=list(df['prediction'])
        lsr.pop(0)
        lsr.append(prediction[0])
        for i in range(2,6):
            
            df['prediction'] = lsr

            
        #df.dropna(inplace=True)

            print("------------------")
            print(df)
            print("i--->",i)
            print("------------------")
            X = np.array(df.drop(['prediction'], 1))
            print(X)
            print(len(X))
            Y = np.array(df['prediction'])
            print(Y)
            print(len(Y))
            X = preprocessing.scale(X)
            X=  preprocessing.normalize(X)
            print(X)
            X_prediction = X[-forecast_time:]
            X_train, Y_train=X[:-1],Y[:-1]
            clf = DecisionTreeRegressor()
            clf.fit(X_train, Y_train)
            prediction = (clf.predict(X_prediction))
            lsr.pop(0)
            lsr.append(prediction[0])
            l.append(prediction[0])
        #     print("Random Forest")
        #     print("Prediction",prediction)
        #     print("ashfj")
        return list(l)
#     print("Dec Tree")
#     print("Prediction",prediction)
#     print("bcnasdb")
##      return list(prediction)

  
#     print("Random Forest")
    if nnn == 2 :

        clf = RandomForestRegressor()
        clf.fit(X_train, Y_train)
        prediction = (clf.predict(X_prediction))


        
        #prediction = (clf.predict(X_prediction))
        l=[prediction[0]]
        lsr=list(df['prediction'])
        lsr.pop(0)
        lsr.append(prediction[0])
        for i in range(2,6):
            
            df['prediction'] = lsr

            
        #df.dropna(inplace=True)

            print("------------------")
            print(df)
            print("i--->",i)
            print("------------------")
            X = np.array(df.drop(['prediction'], 1))
            print(X)
            print(len(X))
            Y = np.array(df['prediction'])
            print(Y)
            print(len(Y))
            X = preprocessing.scale(X)
            X=  preprocessing.normalize(X)
            print(X)
            X_prediction = X[-forecast_time:]
            X_train, Y_train=X[:-1],Y[:-1]
            clf = RandomForestRegressor()
            clf.fit(X_train, Y_train)
            prediction = (clf.predict(X_prediction))
            lsr.pop(0)
            lsr.append(prediction[0])
            l.append(prediction[0])
#     print("Random Forest")
#     print("Prediction",prediction)
#     print("ashfj")
        return list(l)



def predictDataByVolume(stock,days,nnn,labelsStock):
##    end = date.today()
##    #start = datetime(2019, 09, 10)
##    start =  date.today() - relativedelta(days=+15)
    #Outputting the Historical data into a .csv for later use
    #df = get_historical_data(stock, start,output_format='pandas')
    end = date.today()
    start =  date.today() - relativedelta(days=+14)
    start1=date.today() - relativedelta(days=+15)
    flagsD=0
    cntD=0
    cntDE=0
    cntDS=0
    while(flagsD==0):
        try:
            df = data.get_data_yahoo(stock, start, end)
            flagsD=1
            break
        except:
            if 14>cntD>=7:
                cntDE+=1
                end = date.today()- relativedelta(days=+cntDE)
                start =  date.today() - relativedelta(days=+14)
                start1=date.today() - relativedelta(days=+15)
                if cntD==13:
                    cntDE=0
                    cntDS=0
            elif cntDE==0 and cntD<7:
                cntDS+=1
                start =  date.today() - relativedelta(days=+14+cntDS)
                start1=date.today() - relativedelta(days=+15+cntDS)

            else:
                cntDE+=1
                end = date.today()- relativedelta(days=+cntDE)
                cntDS+=1
                start =  date.today() - relativedelta(days=+14+cntDS)
                start1=date.today() - relativedelta(days=+15+cntDS)
            cntD+=1
            
    print(df.head())
    s1=df.index[0]
    e1=df.index[-1]

    s1=str(s1).split(" ")[0]
    e1=str(e1).split(" ")[0]
    
##    s1=str(s1).split(" ")[0]
##    e1=str(e1).split(" ")[0]

    q1=str(labelsStock)#+" "+str(stock).split(".")[0]+" India"
    print(q1)
    df=getNews(df,s1,e1,q1)

#     print(stock)
#     print("before",df.head(1))    
    # csv_name = ('Exports/' + stock + '_Export.csv')    
    # df.to_csv(csv_name)

    print("---------------")
    print(df)
    print("---------------")
    
    df['prediction'] = df['Volume'].shift(-1)
#     print("after",df.head(1))
#     print(df['prediction'][-2])
    print(df.head())
    df.dropna(inplace=True)
    print(df.head())
    forecast_time = int(1)

    X = np.array(df.drop(['prediction'], 1))
    print(X)
    print(len(X))
    Y = np.array(df['prediction'])
    print(Y)
    print(len(Y))
    X = preprocessing.scale(X)
    X=  preprocessing.normalize(X)
    print(X)
    X_prediction = X[-forecast_time:]
    X_train, Y_train=X[:-1],Y[:-1]
        
    print("-----------------")
    print(df)
    print("-----------------")
    


    
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.00000000000000000000000001,shuffle=False)
#     print(X_train)
    #Performing the Regression on the training data
    if nnn == 0:
        clf = LinearRegression()
        clf.fit(X_train, Y_train)
        prediction = (clf.predict(X_prediction))
        
        l=[prediction[0]]
        lsr=list(df['prediction'])
        lsr.pop(0)
        lsr.append(prediction[0])
        for i in range(2,6):
            
            df['prediction'] = lsr

            
        #df.dropna(inplace=True)

            print("------------------")
            print(df)
            print("i--->",i)
            print("------------------")
            X = np.array(df.drop(['prediction'], 1))
            print(X)
            print(len(X))
            Y = np.array(df['prediction'])
            print(Y)
            print(len(Y))
            X = preprocessing.scale(X)
            X=  preprocessing.normalize(X)
            print(X)
            X_prediction = X[-forecast_time:]
            X_train, Y_train=X[:-1],Y[:-1]
            clf = DecisionTreeRegressor()
            clf.fit(X_train, Y_train)
            prediction = (clf.predict(X_prediction))
            lsr.pop(0)
            lsr.append(prediction[0])
            l.append(prediction[0])

  #     print("Linear Regression")
  #     print("Prediction",prediction)
  #     print("hejfhiodhviodjivd")
        return list(l)
    
    
#     print("Dec Tree")
    if nnn == 1 :
        clf = DecisionTreeRegressor()
        clf.fit(X_train, Y_train)
        prediction = (clf.predict(X_prediction))

        l=[prediction[0]]
        lsr=list(df['prediction'])
        lsr.pop(0)
        lsr.append(prediction[0])
        for i in range(2,6):
            
            df['prediction'] = lsr

            
        #df.dropna(inplace=True)

            print("------------------")
            print(df)
            print("i--->",i)
            print("------------------")
            X = np.array(df.drop(['prediction'], 1))
            print(X)
            print(len(X))
            Y = np.array(df['prediction'])
            print(Y)
            print(len(Y))
            X = preprocessing.scale(X)
            X=  preprocessing.normalize(X)
            print(X)
            X_prediction = X[-forecast_time:]
            X_train, Y_train=X[:-1],Y[:-1]
            clf = DecisionTreeRegressor()
            clf.fit(X_train, Y_train)
            prediction = (clf.predict(X_prediction))
            lsr.pop(0)
            lsr.append(prediction[0])
            l.append(prediction[0])
        #     print("Random Forest")
        #     print("Prediction",prediction)
        #     print("ashfj")
        return list(l)
#     print("Dec Tree")
#     print("Prediction",prediction)
#     print("bcnasdb")
##      return list(prediction)

  
#     print("Random Forest")
    if nnn == 2 :

        clf = RandomForestRegressor()
        clf.fit(X_train, Y_train)
        prediction = (clf.predict(X_prediction))


        
        #prediction = (clf.predict(X_prediction))
        l=[prediction[0]]
        lsr=list(df['prediction'])
        lsr.pop(0)
        lsr.append(prediction[0])
        for i in range(2,6):
            
            df['prediction'] = lsr

            
        #df.dropna(inplace=True)

            print("------------------")
            print(df)
            print("i--->",i)
            print("------------------")
            X = np.array(df.drop(['prediction'], 1))
            print(X)
            print(len(X))
            Y = np.array(df['prediction'])
            print(Y)
            print(len(Y))
            X = preprocessing.scale(X)
            X=  preprocessing.normalize(X)
            print(X)
            X_prediction = X[-forecast_time:]
            X_train, Y_train=X[:-1],Y[:-1]
            clf = RandomForestRegressor()
            clf.fit(X_train, Y_train)
            prediction = (clf.predict(X_prediction))
            lsr.pop(0)
            lsr.append(prediction[0])
            l.append(prediction[0])
#     print("Random Forest")
#     print("Prediction",prediction)
#     print("ashfj")
        return list(l)





    
    
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
    
#predictData('AAPL', 5, 1)
#getStocks(5)
