# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:46:34 2025

@author: Mariana Khachatryan
"""

#Load the raw CSV
#Train/test split
#Undersample majority class
#Save processed data as Parquet


from pyspark.sql import functions as F
from src.spark_session import get_spark
from src import config

def load_raw_data(spark):
    return spark.read.csv(config.PAYSIM_CSV, header=True, inferSchema=True)

def train_test_split(df, test_fraction=0.3, seed=config.RANDOM_SEED):
    train_df, test_df = df.randomSplit([1 - test_fraction, test_fraction], seed=seed)
    return train_df, test_df

def make_balanced_train(train_df, label_col=config.LABEL_COL, ratio=config.TRAIN_FRAUD_TO_NONFRAUD_RATIO):
    fraud_df = train_df.filter(F.col(label_col) == 1)
    nonfraud_df = train_df.filter(F.col(label_col) == 0)

    fraud_count = fraud_df.count()
    nonfraud_count = nonfraud_df.count()

    fraction = min(1.0, (fraud_count * ratio) / nonfraud_count)

    nonfraud_sample = nonfraud_df.sample(
        withReplacement=False,
        fraction=fraction,
        seed=config.RANDOM_SEED
    )

    train_balanced = fraud_df.union(nonfraud_sample)
    return train_balanced

def main():
    print(">>> data_prep started")
    spark = get_spark("DataPrep")
    df = load_raw_data(spark)

    train_df, test_df = train_test_split(df)
    train_balanced = make_balanced_train(train_df)

    #train_balanced.write.mode("overwrite").parquet(config.TRAIN_PARQUET)
    
    try:
        train_balanced.write.mode("overwrite").parquet(config.TRAIN_PARQUET)
    except Exception as e:
        print("PYTHON ERROR:", e)
    
    
    test_df.write.mode("overwrite").parquet(config.TEST_PARQUET)
    print(">>> data_prep completed successfully")
    spark.stop()

if __name__ == "__main__":
    main()