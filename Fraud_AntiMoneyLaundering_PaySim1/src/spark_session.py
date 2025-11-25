# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:45:12 2025

@author: Mariana Khachatryan
"""
# Reusable Spark session creation.

from pyspark.sql import SparkSession

def get_spark(app_name: str = "FraudAML") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark