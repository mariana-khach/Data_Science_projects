# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:40:39 2025

@author: Mariana Khachatryan
"""

#All paths + basic hyperparameters in one place.


import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Make Spark-compatible file:// URIs
PROCESSED_URI = f"file:///{PROCESSED_DIR.replace(os.sep, '/')}"

# Files
PAYSIM_CSV = os.path.join(RAW_DIR, "PS_20174392719_1491204439457_log.csv")

TRAIN_PARQUET = f"{PROCESSED_URI}/train_balanced.parquet"
TEST_PARQUET  = f"{PROCESSED_URI}/test.parquet"

# Modeling config
LABEL_COL = "isFraud"
RANDOM_SEED = 42
TRAIN_FRAUD_TO_NONFRAUD_RATIO = 5.0  # undersampling ratio

# RF hyperparams (starting point)
RF_NUM_TREES = 200
RF_MAX_DEPTH = 10
RF_SUBSAMPLE = 0.8
