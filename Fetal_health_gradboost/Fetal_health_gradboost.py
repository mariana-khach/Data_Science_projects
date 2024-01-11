# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:37:07 2023

@author: Mariana Khachatryan
"""


# The goel of this analysis is to predict fetal_helth given the provided data,
# using Gradient_boost and Decision tree machine learning algorithms




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay



df = pd.read_csv("fetal_health.csv")

print(df.info())
print(df.head(10))
print(df.describe())


# %%

# Lets look a correlations in data
data_corr = df.corr()
sns.heatmap(data_corr)

# This shows that fetal health is strongly correlated with accelerations,
# prolonged_decelerations, abnormal_short_term_variability, percentage_of_time_with_abnormal_long_term_variability
# which means we should definitely include these in predictions

# Now lets look at individual distribuions of different variables
df.hist(bins=30, figsize=(15, 20))
# %%

# Target to be predicted 
y = df["fetal_health"]
X = df.drop("fetal_health", axis=1)
#Fetal health can take values 1,2,3


X_tr, X_ts, y_tr, y_ts = train_test_split(X, y)



# We will use classification trees since the response variable is not continuous
# and needs to be splitted into classes
# Using piplines to bundle preprocessing and modeling steps to keep the code organized
pipe_DT = Pipeline([('DT', DecisionTreeClassifier())])
pipe_RF = Pipeline([('RF', RandomForestClassifier())])
pipe_GB = Pipeline([('GB', GradientBoostingClassifier())])



print(list(pipe_DT.named_steps.keys())[0])

pipe_list = [pipe_DT, pipe_GB, pipe_RF]


for i,pipe in enumerate(pipe_list):
    pipe_fit=pipe.fit(X_tr, y_tr)
    print("Score from model "+list(pipe.named_steps.keys())[0]+" is "+str(pipe_fit.score(X_ts,y_ts)))
    print(cross_val_score(pipe, X_tr, y_tr, cv=5,scoring='accuracy'))
    print(i)

 
# The score is the highest for cross validation
# in Gradient Boost methods

#%%Lets try to find best parameters using GridSearchCV

#Do 9 fits with different parameter configurations
parameters={"n_estimators":[200,500,1000],"learning_rate":[0.05,0.5,1]}

my_model = GradientBoostingClassifier()
mod_grid_search=GridSearchCV(my_model,parameters)
mod_grid_search.fit(X_tr, y_tr)

# print best parameter after tuning 
print(mod_grid_search.best_params_) 
grid_predictions = mod_grid_search.predict(X_ts) 



#%%
#Check best parameters

pipe_GB_best = Pipeline([('GB', GradientBoostingClassifier(learning_rate= 0.05, n_estimators=200))])

pipe_GB_best.fit(X_tr, y_tr)
print(cross_val_score(pipe_GB_best, X_tr, y_tr, cv=5,scoring='accuracy'))

best_predict=pipe_GB_best.predict(X_ts)
print("Score from best model "+list(pipe_GB_best.named_steps.keys())[0]+" is "+str(pipe_GB_best.score(X_ts,y_ts)))



#%%

#Lets look at confusion matrix C o check performance of our model, where Cij is equal to the number of 
#observations known to be in group i and predicted to be in group j
conf_mat=confusion_matrix(y_ts, best_predict, labels=pipe_GB_best.classes_, normalize="all")
disp = ConfusionMatrixDisplay( confusion_matrix=conf_mat,display_labels=pipe_GB_best.classes_)
disp.plot()

#we can see that 96% percent of predictions are accurate(sum of diagonal elements)

#For more details on model accuracy we can look at classification report
print(classification_report(y_ts, best_predict)) 
#f1 scores look good as they are close to 1
