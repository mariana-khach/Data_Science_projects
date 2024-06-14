# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:14:24 2024

@author: Mariana Khachatryan
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




class dat:

    def __init__(self,file_path,y_name,ts_size=0.1, rand_state=101):


        """
        We need to supply the path for data file,label name, 
        test data size and random seed to be used in splitting.
        """
        

        self.file_path=file_path
        self.y_name=y_name
        self.df= pd.read_csv(file_path)
        self.X=self.df.drop(y_name,axis=1)
        self.y=self.df[y_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=ts_size,random_state=rand_state)
        
        
        

    def __str__(self):
        return "File path is {}, dependent variable is {}".format(self.file_path,self.y_name)
    
    def scale_dat(self):
        
        scaler=StandardScaler()
        scaler.fit(self.X_train) # fit only raining data to learn its features
        self.X_train=scaler.transform(self.X_train)
        self.X_test=scaler.transform(self.X_test)
        


