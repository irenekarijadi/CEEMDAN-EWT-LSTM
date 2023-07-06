#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyEMD import CEEMDAN
import math
import tensorflow as tf
import numpy
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import metrics
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from math import sqrt

#convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[2]:


##SVR

def svr_model(new_data,i,look_back,data_partition,cap):

    import numpy as np
    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
        
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    
    from sklearn.svm import SVR

    grid = SVR(kernel='rbf')
    grid.fit(X,y)
    y_pred_train_svr= grid.predict(X)
    y_pred_test_svr= grid.predict(X1)

    y_pred_train_svr=pd.DataFrame(y_pred_train_svr)
    y_pred_test_svr=pd.DataFrame(y_pred_test_svr)

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)
 
        
    y_pred_test1_svr= sc_y.inverse_transform (y_pred_test_svr)
    y_pred_train1_svr=sc_y.inverse_transform (y_pred_train_svr)
   
    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)
     
    y_pred_test1_svr=pd.DataFrame(y_pred_test1_svr)
    y_pred_train1_svr=pd.DataFrame(y_pred_train1_svr)
       
    y_test= pd.DataFrame(y_test)
  
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1_svr))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1_svr))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1_svr)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[3]:


##ANN

def ann_model(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)


    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
    
    
    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)
    
    
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.recurrent import LSTM


    neuron=128
    model = Sequential()
    model.add(Dense(units = neuron,input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mse',optimizer=optimizer)

    model.fit(trainX1, y,verbose=0)

    # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test= numpy.array(y_pred_test).ravel()

    y_pred_test=pd.DataFrame(y_pred_test)
    y_pred_test1= sc_y.inverse_transform (y_pred_test)
    y1=pd.DataFrame(y1)
      
    y_test= sc_y.inverse_transform (y1)
       
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[4]:


##RF 
def rf_model(new_data,i,look_back,data_partition,cap):
    
    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    import numpy
    
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    
    from sklearn.ensemble import RandomForestRegressor
    

    grid = RandomForestRegressor()
    grid.fit(X,y)
    y_pred_train_rf= grid.predict(X)
    y_pred_test_rf= grid.predict(X1)

    y_pred_train_rf=pd.DataFrame(y_pred_train_rf)
    y_pred_test_rf=pd.DataFrame(y_pred_test_rf)

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)
 
        
    y_pred_test1_rf= sc_y.inverse_transform (y_pred_test_rf)
    y_pred_train1_rf=sc_y.inverse_transform (y_pred_train_rf)
   
    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)
     
    y_pred_test1_rf=pd.DataFrame(y_pred_test1_rf)
    y_pred_train1_rf=pd.DataFrame(y_pred_train1_rf)
       
    y_test= pd.DataFrame(y_test)
        
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1_rf))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1_rf))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1_rf)

    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[5]:


##LSTM
def lstm_model(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    
    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)


    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
    
    
    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)
    
    
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.recurrent import LSTM


    neuron=128
    model = Sequential()
    model.add(LSTM(units = neuron,input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=optimizer)

    model.fit(trainX1, y, epochs = 100, batch_size = 64,verbose=0)
  # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test= numpy.array(y_pred_test).ravel()

    y_pred_test=pd.DataFrame(y_pred_test)
    y_pred_test1= sc_y.inverse_transform (y_pred_test)
    y1=pd.DataFrame(y1)
      
    y_test= sc_y.inverse_transform (y1)
       
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[6]:


##HYBRID EMD LSTM

def emd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD
    import ewtpy
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers.recurrent import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[7]:


##HYBRID EEMD LSTM

def eemd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EEMD
    import ewtpy
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers.recurrent import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[8]:


##HYBRID CEEMDAN LSTM

def ceemdan_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import CEEMDAN
    
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers.recurrent import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[9]:


##Proposed Method Hybrid CEEMDAN-EWT LSTM

def proposed_method(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=32
    neuron=32
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers.recurrent import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[10]:


##HYBRID EEMD BO LSTM

def eemd_bo_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EEMD
    import ewtpy
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from sklearn.metrics import mean_squared_error
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
        from keras.callbacks import EarlyStopping
        from bayes_opt import BayesianOptimization
        # Define the LSTM model
        def create_model(units,learning_rate):
            model = Sequential()
            model.add(LSTM(units,input_shape=(trainX.shape[1], trainX.shape[2])))
            model.add(Dense(1))
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='mse',optimizer=optimizer)
            return model
        
        # Define the objective function for Bayesian Optimization
        def lstm_cv(units,learning_rate):
            model = create_model(int(units),learning_rate)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
            history = model.fit(trainX, y, epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[es])
            y_pred = model.predict(testX)
            mse = mean_squared_error(y1, y_pred)
            return -mse

        # Define the parameter space for Bayesian Optimization
        pbounds = {'units': (50,200),'learning_rate': (0.001, 0.01)}

        # Run Bayesian Optimization
        lstm_bo = BayesianOptimization(f=lstm_cv, pbounds=pbounds, random_state=42)
        lstm_bo.maximize(init_points=5, n_iter=10, acq='ei')

        # Print the optimal hyperparameters
        print(lstm_bo.max)
    
        opt_lr = lstm_bo.max['params']['learning_rate']
        opt_unit= lstm_bo.max['params']['units']
        opt_units=int(opt_unit)

        neuron=opt_units
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[ ]:


##HYBRID EMD ENN

def emd_enn(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['LV ActivePower (kW)']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD
    import ewtpy
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers import LSTM, Dense,SimpleRNN
        from keras.callbacks import EarlyStopping



        model = Sequential()
        model.add(SimpleRNN(units=neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mse')

        model.fit(trainX, y, epochs = epoch,batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape.to_string())
    print('RMSE',rmse)
    print('MAE',mae)


# In[ ]:




