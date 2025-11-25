fraud-aml-pyspark/
│
├─ data/
│   ├─ raw/
│   │   └─ PS_20174392719_1491204439457_log.csv
│   └─ processed/
│       ├─ train_balanced.parquet
│       ├─ test.parquet
│
├─ notebooks/
│   └─ 01_eda_and_feature_engineering.ipynb
│
├─ src/
│   ├─ config.py
│   ├─ spark_session.py
│   ├─ data_prep.py
│   ├─ features.py
│   ├─ train_supervised.py
│   ├─ evaluate.py
│   └─ utils.py
│
├─ models/
│   ├─ lr_model/
│   └─ rf_model/
│
├─ reports/
│   └─ metrics.json
│
├─ README.md
└─ requirements.txt
