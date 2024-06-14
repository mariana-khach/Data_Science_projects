# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:21:28 2023

@author: Mariana Khachatryan
"""

#Using Linear Regression model for predicting salary based on experince

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



#Reading data file and geting info
sal_dat=pd.read_csv("Salary_dataset.csv")
print(sal_dat.info() )
print("\n")
print( sal_dat.describe())

#%%
x_dat=sal_dat["YearsExperience"].tolist()
y_dat=sal_dat["Salary"].tolist()

#%%
plt.plot(x_dat,y_dat,color="blue",marker="*",linestyle=":")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

# We can see that there is a linear dependence between dependent variable salary 
#and independent variable Years Experience


#There four assumptions associated with linear regression model are
#1. Linearity
#2. Homoscedasticity
#3. Independence
#4. Normality

#%%

# We want to develope a model 
#1. To see if there is a correlatio between salary and years experince.
#2. Predic salary based on experince

# For this we fit our data using Least Square Method



slope, intercept, r_val, p_val, std_err = stats.linregress(x_dat,y_dat)
#p-value for a hypothesis test H0 states that the slope is zero

def F_fit(x):
    return x*slope+intercept

#Calculating fitted values of dependent variable and residuals
y_fit=[F_fit(x) for x in x_dat]
residuals=[y_dat[i]-y_fit[i] for i in range(0,len(y_dat))]


#Plotting fit results with data
plt.plot(x_dat,y_dat,color="blue",marker="*",linestyle=":",label="data",linewidth=1)
plt.plot(x_dat,y_fit,color="red",label="fit",linewidth=1)
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.legend("upper left")
plt.show()

#%%

#Checking Homoscedasticity

plt.plot(y_fit,residuals,"*")
plt.xlabel("Fited values")
plt.ylabel("Residuals")
plt.show()
# We can see that there is no difference between variations in one part of 
#the data and the other

#%%
#Checking Normality

stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

#The points lie approximately at 45 degree reference line, which indicates that
#distribution is normal 

