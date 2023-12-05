# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:25:06 2023

@author: Mariana Khachatryan
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor




iowa_data=pd.read_csv("train.csv")


print(iowa_data.describe())
print(iowa_data.columns)

# y prediction target
y=iowa_data["SalePrice"]

#X : The columns used for predicting price
X=iowa_data[["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]]


#Define decision tree model
dectree_model=DecisionTreeRegressor(random_state=1)

#train the model
dectree_model.fit(X,y)

#make predictions
predictions=dectree_model.predict(X)

print("predicted values are")
print(predictions[:5])
print("true values are")
print(y.head(5))

#Calculating mean absolute error
MAE=mean_absolute_error(y,predictions)

print(MAE)


#%%

# Now let us split the data into two parts to use in training and prediction
X_train,X_dat,y_train,y_dat=train_test_split(X,y,random_state=0)


new_mod=DecisionTreeRegressor(random_state=1)
new_mod.fit(X_train,y_train)
new_pred=new_mod.predict(X_dat)
new_MAE=mean_absolute_error(y_dat,new_pred)
print("MAE from fitting and predictiong with different samples is {} ".format(new_MAE))


#%%
#now lets find the optimum number of leaves, between underfitting and ovefitting

def F_fit(N_leaves,X_t,X_d,y_t,y_d):
    
    mod=DecisionTreeRegressor(max_leaf_nodes=N_leaves,random_state=1)
    mod.fit(X_t,y_t)
    pred_i=mod.predict(X_d)
    MAE_i=mean_absolute_error(y_d,pred_i)
    return (MAE_i)
    

leaves=[10,100,1000,10000]
for i in leaves:
    print(i)
    print(F_fit(i,X_train,X_dat,y_train,y_dat))

#%%
#100 seems reasonable choice for number of leaves
# We can use this to fit entire data

#Define decision tree model
tree_model=DecisionTreeRegressor(max_leaf_nodes=100,random_state=1)

#train the model
tree_model.fit(X,y)

#make predictions
predictions_100=tree_model.predict(X)

print("predicted values are")
print(predictions_100[:5])
print("true values are")
print(y.head(5))

#Calculating mean absolute error
MAE=mean_absolute_error(y,predictions_100)

print(MAE)

#%%

#Now we use random forest model, that uses multiple regression trees and averages
#over them instead of using single decisison tree as was used in regression tree model
randfor_model=RandomForestRegressor(random_state=1)
randfor_model.fit(X_train,y_train)
randfor_predict=randfor_model.predict(X_dat) 

#Calculating mean absolue error for random forest model
randomfor_MAE=mean_absolute_error(y_dat,randfor_predict)

print("MAE from Random Forest model is: {}".format(randomfor_MAE))


