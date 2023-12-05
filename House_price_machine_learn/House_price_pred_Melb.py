# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:36:20 2023

@author: Maiana Khachatryan
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor


train_data=pd.read_csv("train.csv")


print(train_data.describe())

print(train_data.columns)

avg_lot_size=10517
newest_home_age=13



melb_data=pd.read_csv("melb_data.csv")
print(melb_data.columns)

#remove rows wih null values
melb_data=melb_data.dropna(axis=0)

#%%

#Select prediction target
y=melb_data.Price

#Features:the columns that will be used to predict house prices
X=melb_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]



#Defining a decsion tree model and specifying a number for random state
melb_model=DecisionTreeRegressor(random_state=1)

#Training my decsion tree model
melb_model.fit(X,y)


#predicting price for first 10 houses
print ("prediction is:")
print(melb_model.predict(X.head(10)))
print ("true values are:")
y.head(10)

