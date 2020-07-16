from sklearn.externals import joblib
#import package
import pandas_datareader.data as web
#import package
import pandas as pd
import numpy as np
import datetime
from datetime import date
from workalendar.europe import Turkey
#to plot within notebook
import matplotlib.pyplot as plt
import csv
import math
import sys
from workalendar.europe import Turkey
from keras.models import load_model
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
#for normalizing data
from sklearn.preprocessing import MinMaxScaler

#importing required libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from math import sqrt


sc= MinMaxScaler(feature_range=(0,1))
tickers = ["AKBNK.IS","ARCLK.IS" ,"ASELS.IS","BIMAS.IS","GARAN.IS","DOHOL.IS","EKGYO.IS"
            ,"EREGL.IS","FROTO.IS","HALKB.IS","ISCTR.IS","KCHOL.IS"
           ,"KOZAA.IS","KOZAL.IS","KRDMD.IS","PETKM.IS","PGSUS.IS","SAHOL.IS"
           ,"SISE.IS","SODA.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS"
           ,"TOASO.IS","TSKB.IS","TTKOM.IS","TUPRS.IS","VAKBN.IS","YKBNK.IS"]
for k in range(len(tickers)):
    print(tickers[k])
    df = web.DataReader(tickers[k],'yahoo')
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    
    data = df.sort_index(ascending=True, axis=0)
    data['Average']=(data['High']+data['Low'])/2
    input_feature=pd.DataFrame(index=range(0,len(df)),columns=['Date','Average','Volume','Open','Close'])
    for i in range(0,len(data)):
        input_feature['Date'][i]=data['Date'][i]
        input_feature['Average'][i]=data['Average'][i]
        input_feature['Volume'][i]=data['Volume'][i]
        input_feature['Open'][i]=data['Open'][i]
        input_feature['Close'][i]=data['Close'][i]
    input_feature.index=input_feature.Date
    input_feature.drop('Date',axis=1,inplace=True)
    
    
    lookback=50
    test_size=int(.7*len(data))
    
    
    input_data=input_feature
    veri=input_data.values
    dataset=pd.DataFrame(veri)
    train = dataset.iloc[0:test_size,:]
    valid = dataset.iloc[test_size: ,:]
    
    fit_data=np.zeros(shape=(train.shape[0],train.shape[1]))
    col=np.array([])
    for i in range(train.shape[1]):
        col=train.iloc[:,i]
        seri=pd.DataFrame(col)
        fit_data=np.column_stack((fit_data,sc.fit_transform(seri)))
    fit_data=np.delete(fit_data,np.s_[0:4:],axis=1)
    
    x_train,y_train=[],[]
    
    for i in range(lookback,len(train)):
        x_train.append(fit_data[i-lookback:i,])
        y_train.append(fit_data[i,3])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
#    model=load_model('LSTM_AKBANK_Multi2.h5')
    model = Sequential()
    model.add(LSTM(units=30, return_sequences= True, input_shape=(x_train.shape[1],4)))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dense(units=1))
    model.summary()
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x_train, y_train, epochs=60, batch_size=32)
    
    file_name = tickers[k] + '.LSTM.pkl'
#    model.save(file_name)
#    print("Saved model `{}` to disk".format(file_name))
    joblib.dump(model,file_name)
    sonuc={'Date':[],
       'Prediction':[]}
    sonuc=pd.DataFrame(sonuc)
#   sonuc.to_csv (r'C:\Users\betul\Desktop\Project\Web\sonuc.csv', index = None, header=True) 
    
    #
    sonuc=sonuc.append([{'Date':tickers[k],'Prediction':file_name}])
    with open('sonuc.csv','a') as newFile:
        newFileWriter = csv.writer(newFile)
    #    newFileWriter.writerow(['Date','Prediction','Price'])
        newFileWriter.writerow([tickers[k],file_name])
        