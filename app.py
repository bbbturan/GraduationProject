from flask import Flask, jsonify
import csv
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
import numpy as geek 
#import os
from datetime import date
from workalendar.europe import Turkey
#from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
#import pickle
from sklearn.externals import joblib
#to plot within notebook
#import package
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
#for normalizing data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#importing required libraries
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM
#from math import sqrt

app = Flask(__name__)
#@app.route('/api/stockPrices/<string:stock_code>',methods=['GET'])
#def get_stockPrice(stock_code):
#    with open("sonuc.csv","r") as csvFile:
#        csvFileReader=csv.reader(csvFile)
#        for row in csvFileReader:
#            for field in row:
#                if(field==stock_code):
#                    print(row)
#                    return jsonify({'sonuc:' : row[1]})
#                
#    csvFile.close()
    
    
    
@app.route('/api/stockPrices/<string:stock_code>',methods=['GET'])
def get_stockPrice(stock_code):
    with open("sonuc.csv","r") as csvFile:
        csvFileReader=csv.reader(csvFile)
        for row in csvFileReader:
            for field in row:
                if(field==stock_code):
                    print(row)
                    df = web.DataReader(row[0],'yahoo')
                    sc= MinMaxScaler(feature_range=(0,1))
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
    
                    an = datetime.datetime.now()
                    tarih=datetime.datetime.strftime(an, '%X')
                    
                    if tarih < '18:05:00':
                        last_date=df['Date'].iloc[-2]
                        last_fifth_days=input_feature[-51:-1]
                    else:
                        last_date=df['Date'].iloc[-1]
                        last_fifth_days=input_feature[-50:]
                        
                    print(last_date)
                    next_date= last_date + datetime.timedelta(days=1)
                    cal=Turkey()
                    #
                    def control_date(ndate):
                        result=cal.is_working_day(date(next_date.year,next_date.month,next_date.day))
                        return result
                    while control_date(next_date)==False:
                        next_date= next_date + datetime.timedelta(days=1)
                        control_date(next_date)
                    
                    print(next_date)
                    #
                    #ten_days=sc.transform(last_fifth_days)
#                    model=load_model(row[1])
#                    model=pickle.load(row[1])
                    model_path='C:/Users/betul/Desktop/Betül/ders/4.Sınıf/Bitirme Projesi/stockPriceProject/' + row[1]
                    print(model_path)
                    with open(model_path,'rb') as f:
                        model=joblib.load(f)
#                    model=pickle.dump(open(model_path,'rb'))
                    print(type(model))
                    print(model)
                    fifth_days=np.zeros(shape=(last_fifth_days.shape[0],last_fifth_days.shape[1]))
                    col3=np.array([])
                    for i in range(last_fifth_days.shape[1]):
                        col3=last_fifth_days.iloc[:,i]
                        fifth_valid=pd.DataFrame(col3)
                        fifth_days=np.column_stack((fifth_days,sc.fit_transform(fifth_valid)))
                    fifth_days=np.delete(fifth_days,np.s_[0:4:],axis=1)
                    
                    
                    #close_value=last_fifth_days.iloc[3]
                    #data_close=np.array(close_value)
                    #data_close = np.reshape(data_close, (data_close.shape[0],1))
                    
                    fifth_days = np.reshape(fifth_days,(1,len(fifth_days),4))
                    predicted_date=model.predict(fifth_days)
                    print(predicted_date)
                    predicted_date = sc.inverse_transform(predicted_date)
                    print(predicted_date)
                    print(type(predicted_date))
                    predict=geek.array_str(predicted_date)
                    print(type(predicted_date))
                    return jsonify({'sonuc:' : predict})