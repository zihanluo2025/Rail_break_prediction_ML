# Databricks notebook source
# MAGIC %md 
# MAGIC <div style="padding-top: 10px;  padding-bottom: 10px;">
# MAGIC   <img src="https://insightfactoryai.sharepoint.com/:i:/r/sites/insightfactory.ai/Shared%20Documents/E.%20Marketing/Company%20Logos%20and%20Style%20Guide/PNG/insightfactory.ai%20logo%20multiline%20reversed.png" alt='insightfactory.ai' width=150   style="display: block; margin: 0 auto" /> 
# MAGIC </div>
# MAGIC
# MAGIC # Model Inference
# MAGIC
# MAGIC ***Summary of process:*** This notebook is used to do inference on the ML model from the engineered features.
# MAGIC
# MAGIC ***Input Tables:***  
# MAGIC - 
# MAGIC
# MAGIC ***Output Table:*** 
# MAGIC - 
# MAGIC
# MAGIC Note: This is just the overview of the process, For details please review the notebook thoroughly
# MAGIC
# MAGIC **Business Rules:** <br/>
# MAGIC \<Describe the Business Rules that are encapsulated in this Enrichment\>
# MAGIC
# MAGIC **Dependencies:**<br/>
# MAGIC -
# MAGIC
# MAGIC **Ownership:**<br/>
# MAGIC \<Indicate who owns this Enrichment ruleset\>

# COMMAND ----------

# MAGIC  %md 
# MAGIC
# MAGIC #### Modification Schedule
# MAGIC
# MAGIC | Date | Who | Description |
# MAGIC | ---: | :--- | :--- |
# MAGIC | 2025-08-30 | Sheng Wang | Initial version. |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks] databricks-feature-engineering
# MAGIC dbutils.library.restartPython()
# MAGIC %pip install -U scikit-learn
# MAGIC %pip install -U "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %run "/InsightFactory/Helpers/ML Inference (Unity Catalog) Entry"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook Start
# MAGIC
# MAGIC ### Input Parameters
# MAGIC
# MAGIC All Notebook Parameters (if any) are contained in the dictionary variable 'params'.  There are two ways to get the individual parameter from params, in both cases the parameter name is case-sensitive:
# MAGIC
# MAGIC   1) Use dot-notation - refer to the example below.
# MAGIC
# MAGIC       params = { "Name": "Test", "Values": { "Title": "Results", "Results": [ { "Definition": "Core Sample", "Outcome": "Prospective" }, { "Definition": "Follow-up", "Outcome": "For review" } ] } }
# MAGIC
# MAGIC       params.Name produces 'Test'<br/>
# MAGIC       params.Values.Title produces 'Results'<br/>
# MAGIC       params.Values.Results[0] produces { "Definition": "Core Sample", "Outcome": "Prospective" }<br/>
# MAGIC       params.Values.Results[1].Definition produces 'Follow-up'
# MAGIC
# MAGIC   2) Use the search_dictionary function as follows:  var1 = search_dictionary(params, "parameter-name").  
# MAGIC
# MAGIC       There is an optional third parameter to this function: value_to_return_if_not_found -  this is the value to return if the particular parameter is not found in params.<br/>
# MAGIC       **Note** that value_to_return_if_not_found can take on any type (string, int, boolean, struct, ..) e.g search_dictionary(params, "IncorrectlyNamedParameter", False) will return the boolean False if "IncorrectlyNamedParameter" is not found in params.
# MAGIC
# MAGIC   3) Use the load_pickle_model function as follows: model = load_pickle_model(client, "model_location", "model_version").
# MAGIC
# MAGIC       There is an optional fourth parameter to this function: pickle_location - The exact location of the pickle file in the particular storage volume.<br/>
# MAGIC       **Note** client is the MlFlowClient object and model location is the exact location of the model with 3 part schema as "Catalog.Schema.ModelName"
# MAGIC
# MAGIC
# MAGIC **CAUTION:** There is another dictionary variable, 'config', that contains all of the configuration sent to this Notebook.  In most cases, you will have no use for 'config' but if you choose to use 'config' in this Notebook, note the following:
# MAGIC - Access the individual parameters within config by using the search_dictionary function e.g. search_dictionary(config, "ParameterName").  Dot-notation access **does not apply** to 'config'.
# MAGIC - Heed this **WARNING** - The individual parameter names within 'config' are subject to change outside of your control which may break your code.
# MAGIC <br/><br/>
# MAGIC
# MAGIC ### Enrichment Results
# MAGIC
# MAGIC Add the code you need to perform your enrichment/extract in cell(s) below until the 'Notebook End' cell.
# MAGIC
# MAGIC ####Important: 
# MAGIC - Ensure that the result is stored in a PySpark dataframe as
# MAGIC     - 'df_result' e.g. df_result = ... for the Inference Results for the Table 
# MAGIC     - 'df_inference_data' e.g. df_inference_data = ... for us to perform data at the exit of the model so you don't have to perform inference on model.
# MAGIC <br/><br/>
# MAGIC
# MAGIC ### Running this Notebook directly in Databricks
# MAGIC
# MAGIC This Notebook can be run directly from your Databricks Workspace.  If the Notebook relies on Notebook Parameters, please read the following instructions:
# MAGIC 1) Add this line of code to a cell at the top of your Notebook and run that cell.<br/>
# MAGIC    ```dbutils.widgets.text('ParametersJSON', '{ "ModelName":"name","ModelAlias":"alias","ModelVersion":"version","ModelSchema":"DatabaseName for Model", "NotebookParameters": { "param1": "value1", "param2": "value2" } }')```
# MAGIC 2) This will add a Parameter to the Notebook.  Simply replace (or remove) the pre-canned parameters, 'param1', 'param2' and their values with your own.
# MAGIC 3) When you have finished running this Notebook directly in Databricks, comment out the line of code you added or delete the cell entirely.    

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


TABLE = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.feature_selection.preprocess_predict_table"

def to_date_any(col):
    return F.coalesce(
        F.to_date(col),
        F.to_date(col, "yyyy-MM-dd"),
        F.to_date(col, "yyyy-M-d"),
        F.to_date(col, "yyyy/MM/dd"),
        F.to_date(col, "yyyy/M/d"),
        F.to_date(col, "MM/dd/yyyy"),
        F.to_date(col, "dd/MM/yyyy")
    )

df = spark.table(TABLE).withColumn("Tc_date", to_date_any(F.col("Tc_r_date")))

numeric_types = ("double", "float", "int", "bigint", "decimal")
feature_cols = [
    c for c, t in df.dtypes
    if c.startswith("Wagon_")
    and c != "Wagon_RecordingDate"
    and any(tp in t.lower() for tp in numeric_types)
]
print(f"Number of features: {len(feature_cols)}")

df = df.withColumn(
    "sample_id",
    F.concat_ws("_", F.col("Tc_BaseCode"), F.col("Tc_SectionBreakStartKM"), F.col("Tc_date"))
)

for c in feature_cols:
    df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

# agg_exprs = [F.avg(c).alias(c) for c in feature_cols]
agg_exprs = []
for c in feature_cols:
    agg_exprs.append(F.avg(c).alias(c))

    # if "Acc" in c:
        # agg_exprs.append(F.min(c).alias(f"{c}_min"))
        # agg_exprs.append(F.max(c).alias(f"{c}_max"))
        # agg_exprs.append(F.stddev(c).alias(f"{c}_std"))

test_df = (
    df.groupBy(
        "Tc_BaseCode",
        "Tc_BaseCode_Mapped",
        "Tc_SectionBreakStartKM",
        "Tc_date",
        "sample_id",
        "Tc_p_key",
    )
    .agg(*agg_exprs)
)

feature_cols = [c for c in test_df.columns if c.startswith("Wagon_")]

sample_cols = ( ["Tc_BaseCode", "Tc_BaseCode_Mapped","Tc_SectionBreakStartKM","Tc_date", "sample_id", "Tc_p_key"] + feature_cols[::] ) 


print(f"Row count after aggregation: {test_df.count():,}")
print(f"Column count after aggregation: {len(test_df.columns)}")

all_features = feature_cols
print(f"all features count: {len(all_features)}")

display(test_df.select(*sample_cols).limit(10))


# COMMAND ----------

## Define the data and features that you want to use for the test context
### predictive_maintenance.testcontext contains the records on which you need to predict for evaluating your model and be on the Leader Board
# input_data=spark.sql('''
#       select
#         tc.p_key,
#         w.Twist14m,
#         w.BounceFrt,
#         w.BrakeCylinder
#       from
#         dev_adlunise.predictive_maintenance_uofa_2025.wagondata w
#         join dev_adlunise.predictive_maintenance_uofa_2025.testcontext as tc
#           on concat(w.BaseCode, '_', w.SectionBreakStartKM, '_20m_', w.RecordingDate) = tc.p_key
#     ''').toPandas().fillna(0)

final_feature_cols = [
    col for col in all_features
    if col not in [
        'sample_id', '_seq_position', 'Tc_target',
        'has_last_fail', 'days_since_last_fail', 'Wagon_ICWVehicle',
        'has_data', 'doy_cos', 'doy_sin', 'time_position', 'days_to_target'
    ]
]

cols_for_inference = ["Tc_p_key"] + final_feature_cols


test_input_pdf = (
    test_df.select("*") 
    .toPandas()
    .fillna(0)
)

display(test_input_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load your model
# MAGIC
# MAGIC Refer to mlflow docs online for right model type if run into errors

# COMMAND ----------

import mlflow
# model=mlflow.sklearn.load_model(model_uri)
print(model_uri)
model = mlflow.pyfunc.load_model(model_uri)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform inference and save the results of the inference with p_key and target

# COMMAND ----------

import pandas as pd
# Make predictions on the test set
# y_pred = model.predict(input_data[model.feature_names_in_])

# # Get probability of positive case if available, otherwise use 0.5
# if hasattr(model, 'predict_proba'):
#     y_proba = model.predict_proba(input_data[model.feature_names_in_])
#     # For binary classification, get probability of positive class (index 1)
#     prob_positive = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
# else:
#     # Default to 0.5 if model doesn't output probabilities
#     prob_positive = [0.5] * len(y_pred)

# prediction = pd.DataFrame(
#     {
#         "p_key": input_data['p_key'],
#         "target": y_pred,
#         "probability": prob_positive
#     }
# )

# df_result = spark.createDataFrame(prediction)


prediction = model.predict(test_input_pdf)  
df_result = spark.createDataFrame(prediction)
display(df_result)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook End
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**
# MAGIC
# MAGIC ####Important: 
# MAGIC 1) Ensure that the result is stored in a PySpark dataframe as
# MAGIC  - 'df_inference_data' e.g. df_inference_data = ... for Sending Inference data and performing inference on that.
# MAGIC  - 'df_result' e.g. df_result = ...  for Sending Results of the Inference performed by you.

# COMMAND ----------

# MAGIC %run "/InsightFactory/Helpers/ML Inference (Unity Catalog) Exit"

# COMMAND ----------

# MAGIC %md # Testing or Debugging Zone

# COMMAND ----------

# MAGIC %sql
# MAGIC -- select * from tmp_tc_demo.inferenceSubmission
