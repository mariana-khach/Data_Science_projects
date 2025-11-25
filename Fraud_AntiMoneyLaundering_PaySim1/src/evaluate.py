# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:58:48 2025

@author: Mariana Khachatryan
"""


#Load test set
#Apply same feature pipeline (from saved pipeline model)
#Compute ROC-AUC, PR-AUC, confusion matrix
#Optionally save metrics to reports/metrics.json



import os, json
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from src.spark_session import get_spark
from src.features import add_engineered_columns
from src import config

def load_classifier(model_name):
    path = os.path.join(config.MODELS_DIR, model_name)
    if model_name.startswith("lr"):
        return LogisticRegressionModel.load(path)
    return RandomForestClassificationModel.load(path)

def evaluate_model(model_name="rf_model"):
    spark = get_spark("Evaluate")

    # 1) Load clean test data
    test_df = spark.read.parquet(config.TEST_PARQUET)

    # 2) Add engineered columns ONCE
    test_df = add_engineered_columns(test_df)

    # 3) Load feature model and transform ONCE
    fe_path = os.path.join(config.MODELS_DIR, "feature_model")
    feature_model = PipelineModel.load(fe_path)
    test_fe = feature_model.transform(test_df)

    # 4) Load classifier and predict ONCE
    clf = load_classifier(model_name)
    preds = clf.transform(test_fe)

    # Metrics
    roc_eval = BinaryClassificationEvaluator(
        labelCol=config.LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    pr_eval = BinaryClassificationEvaluator(
        labelCol=config.LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    roc_auc = roc_eval.evaluate(preds)
    pr_auc = pr_eval.evaluate(preds)

    # Confusion matrix
    preds.groupBy(config.LABEL_COL, "prediction").count().show()

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(config.REPORTS_DIR, f"{model_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"roc_auc": roc_auc, "pr_auc": pr_auc}, f, indent=2)

    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    print(f"{model_name} PR-AUC: {pr_auc:.4f}")
    print("Saved:", out_path)

    spark.stop()

if __name__ == "__main__":
    evaluate_model("rf_model")