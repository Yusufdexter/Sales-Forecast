# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:43:27 2020

@author: Yusufdexter
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset


dataset = pd.read_csv('monthly-sales.csv')

dataset.info()
dataset.isnull().sum()
dataset.isnull().sum().values.sum()
dataset.nunique()
dataset.describe()


# Convert Month column to datetime
dataset['Month'] = pd.to_datetime(dataset['Month'])

# Rename column
dataset.columns = ['Month', 'Sales']

# Indexing
dataset = dataset.set_index('Month')

dataset.plot(figsize = (16, 8))
plt.title('Monthly Sales from JAN 2007 - JUNE 2020')
plt.savefig('01 Monthly Sales from JAN 2007 - JUNE 2020.png')


# SMA (Simple Moving Average)
# Create New Columns
dataset['12-Months SMA (mean)'] = dataset['Sales'].rolling(window = 12).mean()
dataset['12-Months SMA (std)'] = dataset['Sales'].rolling(window = 12).std()

# Visualizing (Simple Moving Average)
dataset.plot(figsize = (16, 8))
plt.title('Monthly Sales (Simple Moving Average)')
plt.savefig('02 Monthly Sales (Simple Moving Average).png')


# EWMA (EXPONENTIALLY WEIGHTED MOVING AVERAGE)
# Create New Columns
dataset['12-Months EXPONENTIALLY WEIGHTED MOVING AVERAGE'] = dataset['Sales'].ewm(span = 12).mean()

# Visualising EXPONENTIALLY WEIGHTED MOVING AVERAGE
ewma = dataset[['Sales', '12-Months EXPONENTIALLY WEIGHTED MOVING AVERAGE']]
ewma.plot(figsize = (16, 8))
plt.title('12-Months Exponentially Weighted Moving Average')
plt.savefig('03 Exponentially Weighted Moving Average.png')


# ETS (ERROR - TREND - SEASONALITY)
model = seasonal_decompose(dataset['Sales'], model = 'additive')
fig = model.plot()
plt.savefig('04 Error - Trend - Seasonality.png')


'''
    We can use an additive model when it seems that the trend is more linear and the seasonality and 
    trend components seem to be constant over time (e.g. every year we add 10,000 passengers). 
    A multiplicative model is more appropriate when we are increasing (or decreasing) at a non-linear 
    rate (e.g. each year we double the amount of passengers).
'''




# TESTING FOR STATIONARITY
''' 
    We can use the Augmented Dickey-Fuller unit root test.

    In statistics and econometrics, an augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root 
    is present in a time series sample. The alternative hypothesis is different depending on which version of the test
    is used, but is usually stationarity or trend-stationarity.
'''

model = adfuller(dataset['Sales'])


def adf_check(time_series):
    model = adfuller(time_series)
    print('Augmented Dicky-Fuller Test')
    labels = ['ADF Test Statistic', 'p-value', 'Num of Lags', 'Num of Observation used']
    
    for value, label in zip(model, labels):
        print(f'{label} : {str(value)}')
    
    if model[1] <= 0.05:
        print('Strong evidence against null hypothesis')
        print('reject null hypothesis')
        print('Data has no unit root and is stationary (Not Seasonal)')
    else:
        print('weak evidence against null hypothesis')
        print('Accept null hypothesis')
        print('Data has a unit root, it is non-stationary (Seasonal)')

adf_check(dataset['Sales'])


'''
Important Note!

** We have now realized that our data is seasonal (it is also pretty obvious from the plot itself). This means 
we need to use Seasonal ARIMA on our model. If our data was not seasonal, it means we could use just ARIMA on it. 
We will take this into account when differencing our data! Typically financial stock data won't be seasonal, but 
that is kind of the point of this section, to show you common methods, that won't work well on stock finance data!**
'''


# DIFFERENCING

# 1st First Difference
dataset['First Difference'] = dataset['Sales'] - dataset['Sales'].shift(1)

# Check If Seasonal or Not & Droping Nan value
adf_check(dataset['First Difference'].dropna())

# Visualising
dataset['First Difference'].plot(figsize = (10, 5))
plt.title('First Difference Sales')
plt.annotate('p-value : 0.15243939914373233; Weak evidence against null hypothesis; Accept null hypothesis', 
             (0, 0), (20, -30), fontsize = 10, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.savefig('05 First Difference Sales.png')




# 2nd Second Difference (When Neccesary (if first diffence is still seasonal))
dataset['Second Difference'] = dataset['First Difference'] - dataset['First Difference'].shift(1)

# Check If Seasonal or Not & Droping Nan value
adf_check(dataset['Second Difference'].dropna())

# Visualising
dataset['Second Difference'].plot(figsize = (10, 5))
plt.title('Second Difference Sales')
plt.annotate('p-value : 3.1227365805570227e-27; Strong evidence against null hypothesis; Reject null hypothesis', 
             (0, 0), (20, -30), fontsize = 10, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.savefig('06 Second Difference Sales.png')



# Seasonal Difference
dataset['Seasonal Difference'] = dataset['Sales'] - dataset['Sales'].shift(12)

# Check If Seasonal or Not & Droping Nan value
adf_check(dataset['Seasonal Difference'].dropna()) # Seasonal Difference by itself was not enough!

# Visualising
dataset['Seasonal Difference'].plot(figsize = (10, 5))
plt.title('Seasonal Difference Sales')
plt.annotate('p-value : 0.22472206446714477; Weak evidence against null hypothesis; Accept null hypothesis', 
             (0, 0), (20, -30), fontsize = 10, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.savefig('07 Seasonal Difference Sales.png')


# You can also do Seasonal 1st First Difference
dataset['Seasonal First Difference'] = dataset['First Difference'] - dataset['First Difference'].shift(12)

# Check If Seasonal or Not & Droping Nan value
adf_check(dataset['Seasonal First Difference'].dropna()) # Seasonal Difference by itself was not enough!

# Visualising
dataset['Seasonal First Difference'].plot(figsize = (10, 5))
plt.title('Seasonal Difference Sales')
plt.annotate('p-value : 0.00031129393040702587; Strong evidence against null hypothesis; Reject null hypothesis', 
             (0, 0), (20, -30), fontsize = 10, xycoords = 'axes fraction', textcoords = 'offset points', va = 'top')
plt.savefig('08 Seasonal First Difference Sales.png')



'''
    ACF
'''

# Autocorrelation of first difference
max_lags = dataset['First Difference'].count() - dataset['First Difference'].isnull().sum() # Calculating lags

# Visualising
fig_first = plot_acf(dataset['First Difference'].dropna(), lags = 160) 
plt.title('First Difference ACF')
plt.savefig('09 First Difference ACF.png')

# Alternatively
autocorrelation_plot(dataset['First Difference'].dropna())
plt.title('First Difference ACF (pandas plot)')
plt.savefig('09.1 First Difference ACF.png')

# Autocorrelation of Seasonal first difference
max_lags = dataset['Seasonal First Difference'].count() - dataset['Seasonal First Difference'].isnull().sum() # Calculating lags

# Visualising
fig_seasonal_first = plot_acf(dataset["Seasonal First Difference"].dropna(), lags = 136)
plt.title('Seasonal First Difference ACF')
plt.savefig('10 Seasonal First Difference ACF.png')

# Alternatively
autocorrelation_plot(dataset['Seasonal First Difference'].dropna())
plt.title('Seasonal First Difference ACF (pandas plot)')
plt.savefig('10.1 Seasonal First Difference ACF.png')



'''
    PACF
'''
# Partial Autocorrelation
Max_lags = dataset['Seasonal First Difference'].count() - dataset['Seasonal First Difference'].isnull().sum() # Calculating lags

# Visualising
result = plot_pacf(dataset["Seasonal First Difference"].dropna(), lags = 136, method='ywmle')
plt.title('Seasonal First Difference PACF')
plt.savefig('11 Seasonal First Difference PACF.png')


# Final ACF and PACF Plots
fig = plt.figure(figsize = (12, 8))
ax1 = fig.add_subplot(211)
fig = plot_acf(dataset['Seasonal First Difference'].iloc[13:], lags = 80, ax = ax1, title = 'Seasonal First Difference ACF')
ax2 = fig.add_subplot(212)
fig = plot_pacf(dataset['Seasonal First Difference'].iloc[13:], lags = 80, ax = ax2, method='ywmle', title = 'Seasonal First Difference PACF')
plt.savefig('12 Seasonal First Difference ACF & PACF.png')



'''
    ARIMA model
'''
# Fitting the model
model = sm.tsa.statespace.SARIMAX(dataset['Sales'], order = (0, 1, 0), seasonal_order = (1, 1, 1, 12))
results = model.fit()
print(results.summary())

# Prediction result
result = results.resid

# Visualising
result.plot()
plt.title('Error Points')
plt.show()
plt.savefig('13 Error Points.png')


# Density
result.plot(kind = 'kde')
plt.title('Residual Density')
plt.show()
plt.savefig('14 Residual Density.png')



# Prediction of Future Values
dataset['forecast'] = results.predict(start = 140, end = 174, dynamic = True) 

# Visualising 
dataset[['Sales', 'forecast']].plot(figsize = (12, 8), title = 'Prediction Chart')



#===================
# ARIMA ALTERNATIVE
#===================
future_dates = [dataset.index[-1] + DateOffset(months = x) for x in range(0, 24)]

# Creating future date data
future_dates_dataset = pd.DataFrame(index = future_dates[1:], columns = dataset.columns)
# concatenating
future_dataset = pd.concat([dataset, future_dates_dataset])


# Prediction of Future Values
future_dataset['forecast'] = results.predict(start = 140, end = 174, dynamic = True) 

# Visualising 
future_dataset[['Sales', 'forecast']].plot(figsize = (12, 8)) 
plt.title('12 Months Sales Forecast')
plt.savefig('15 Sales Forecast.png')
