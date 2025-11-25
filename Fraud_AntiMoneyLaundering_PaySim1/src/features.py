# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:52:49 2025

@author: Mariana Khachatryan
"""

#Build the feature pipeline (indexing + encoding + numeric features).
#This keeps all feature logic in one module, so both training & evaluation use the same pipeline.


from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from src import config

ENGINEERED_COLS = ["deltaOrig", "deltaDest", "flagOrigMismatch", "flagDestMismatch"]
PIPELINE_OUTPUT_COLS = ["type_idx", "type_ohe", "features"]

def add_engineered_columns(df):
    return (
        df.withColumn(
            "deltaOrig",
            F.col("oldbalanceOrg") - F.col("newbalanceOrig")
        )
        .withColumn(
            "deltaDest",
            F.col("newbalanceDest") - F.col("oldbalanceDest")
        )
        .withColumn(
            "flagOrigMismatch",
            F.when(
                F.abs(F.col("oldbalanceOrg") + F.col("amount") - F.col("newbalanceOrig")) > 1e-6,
                1
            ).otherwise(0)
        )
        .withColumn(
            "flagDestMismatch",
            F.when(
                F.abs(F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest")) > 1e-6,
                1
            ).otherwise(0)
        )
    )

def build_feature_pipeline():
    type_indexer = StringIndexer(
        inputCol="type",
        outputCol="type_idx",
        handleInvalid="keep"
    )

    type_encoder = OneHotEncoder(
        inputCols=["type_idx"],
        outputCols=["type_ohe"]
    )

    numeric_cols = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "deltaOrig", "deltaDest",
        "flagOrigMismatch", "flagDestMismatch"
    ]

    assembler = VectorAssembler(
        inputCols=numeric_cols + ["type_ohe"],
        outputCol="features"
    )

    return Pipeline(stages=[type_indexer, type_encoder, assembler])