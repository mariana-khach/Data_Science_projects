# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:57:01 2025

@author: 17573
"""

#training script: loads processed data, builds pipeline, trains LR and RF, saves models.
#We can also choose to save fe_model separately and reuse it in evaluate.py, but bundling it in the pipeline is simpler

import os
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from src.spark_session import get_spark
from src.features import add_engineered_columns, build_feature_pipeline
from src.utils import add_class_weights
from src import config

def train_models():
    spark = get_spark("TrainSupervised")

    # 1) Load clean train data
    train_df = spark.read.parquet(config.TRAIN_PARQUET)

    # 2) Add engineered columns ONCE
    train_df = add_engineered_columns(train_df)

    # 3) Fit feature pipeline ONCE
    feature_pipeline = build_feature_pipeline()
    feature_model = feature_pipeline.fit(train_df)

    # 4) Transform ONCE
    train_fe = feature_model.transform(train_df)

    # 5) Add class weights ONCE
    train_w = add_class_weights(train_fe, label_col=config.LABEL_COL)

    # 6) Models (no feature stages here because train_fe already has "features")
    lr = LogisticRegression(
        labelCol=config.LABEL_COL,
        featuresCol="features",
        weightCol="classWeight",
        maxIter=50,
        regParam=0.01
    )

    rf = RandomForestClassifier(
        labelCol=config.LABEL_COL,
        featuresCol="features",
        numTrees=config.RF_NUM_TREES,
        maxDepth=config.RF_MAX_DEPTH,
        subsamplingRate=config.RF_SUBSAMPLE
    )

    lr_model = lr.fit(train_w)
    rf_model = rf.fit(train_w)

    # 7) Save both the feature_model and classifiers
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    feature_model.write().overwrite().save(os.path.join(config.MODELS_DIR, "feature_model"))
    lr_model.write().overwrite().save(os.path.join(config.MODELS_DIR, "lr_model"))
    rf_model.write().overwrite().save(os.path.join(config.MODELS_DIR, "rf_model"))

    spark.stop()

if __name__ == "__main__":
    train_models()