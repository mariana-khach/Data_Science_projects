# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:54:48 2025

@author: Mariana Khachatryan
"""

#Helpers like class weights & basic logging.


from pyspark.sql import functions as F

def add_class_weights(df, label_col: str, total_col: str = "classWeight"):
    total = df.count()
    fraud_total = df.filter(F.col(label_col) == 1).count()
    nonfraud_total = total - fraud_total

    w_fraud = total / (2.0 * fraud_total)
    w_nonfraud = total / (2.0 * nonfraud_total)

    df_w = df.withColumn(
        total_col,
        F.when(F.col(label_col) == 1, w_fraud).otherwise(w_nonfraud)
    )
    return df_w