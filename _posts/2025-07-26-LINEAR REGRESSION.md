---
title: linear regression
categories: [matchine learning]

tags : linear regression, matchine learning


---
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')
df

from google.colab import files
uploaded = files.upload()

%matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

xvaluesdf = df.drop('price',axis='columns')
xvaluesdf



yvaluesdf = df.price
yvaluesdf

# Create linear regression model object
reg = linear_model.LinearRegression()
reg.fit(xvaluesdf.values,yvaluesdf.values)

reg.predict([[3300]])

reg.coef_

reg.intercept_

y=3300*135.78767123 + 180616.43835616432
y

reg.predict([[5000]])

reg.predict([[2000]])

from google.colab import files
uploaded = files.upload()

area_df = pd.read_csv("areas.csv")
area_df.head()

prices_df = reg.predict(area_df.values)
prices_df

area_df['prices']= prices_df
area_df



area_df.to_csv('prediction.csv')

from google.colab import files
uploaded = files.upload()

dfm = pd.read_csv('homeprices-m.csv')
dfm

dfm.bedrooms.median()

dfm.bedrooms = dfm.bedrooms.fillna(dfm.bedrooms.median())
dfm

xvaluesdfm = dfm.drop('price',axis='columns')
xvaluesdfm


mreg = linear_model.LinearRegression()
mreg.fit(xvaluesdfm.values,dfm.price.values)

mreg.coef_

mreg.intercept_

112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540384

mreg.predict([[3000, 3, 40]])



from google.colab import drive
drive.mount('/content/drive')

area_df.to_csv('/content/drive/My Drive/prediction.csv')