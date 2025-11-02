# Databricks notebook source
# MAGIC %md 
# MAGIC <div style="padding-top: 10px;  padding-bottom: 10px;">
# MAGIC   <img src="https://insightfactoryai.sharepoint.com/:i:/r/sites/insightfactory.ai/Shared%20Documents/E.%20Marketing/Company%20Logos%20and%20Style%20Guide/PNG/insightfactory.ai%20logo%20multiline%20reversed.png" alt='insightfactory.ai' width=150   style="display: block; margin: 0 auto" /> 
# MAGIC </div>
# MAGIC
# MAGIC # Model Build
# MAGIC
# MAGIC ***Summary of process:*** This notebook is used to Build the ML model from the engineered features. This results in table with performance of the model and version of the resulting model. It is also preferred to Match your resulting model version with the used pipeline version in the resultant. 
# MAGIC
# MAGIC ***Input Tables:***  
# MAGIC - 
# MAGIC
# MAGIC ***Output Table:*** 
# MAGIC - model_config
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
# MAGIC | 2025-09-16 | Tao Xu | Initial version. |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "/InsightFactory/Helpers/ML Build (Unity Catalog) Entry"

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
# MAGIC     - 'df_result' e.g. df_result = ...  containing Model Name, Model Version, Performance metrics, Pipeline version (for feature log) and model
# MAGIC     - Ensure that your models are registered under ml_catalog. Always name your model as `f'{ml_catalog}.{delta_schema_name}.{model_name}'`
# MAGIC
# MAGIC This will result in model and model config Table in your shared ml_catalog for easy sharing and inference across environments.
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

# MAGIC %md ## import python modules

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, classification_report, precision_recall_curve
)

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

TABLE_NAME = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.rebalanced_tables.random_oversample_fe_training"
RAND_SEED  = 42
ROW_LIMIT  = 200000

def group_from_pkey(s):
    parts = str(s).split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else str(s)

def build_groups(df: pd.DataFrame) -> pd.Series:
    if "p_key" in df.columns:
        return df["p_key"].map(group_from_pkey)
    for combo in [["Tc_BaseCode","Tc_SectionBreakStartKM"],
                  ["Tc_BaseCode","Tc_BaseCode_Mapped","Tc_SectionBreakStartKM"]]:
        if all(c in df.columns for c in combo):
            return df[combo].astype(str).agg("_".join, axis=1)
    return pd.Series(["__nogroup__"] * len(df))

# COMMAND ----------

df_spark = spark.table(TABLE_NAME).orderBy(F.rand(RAND_SEED)).limit(ROW_LIMIT)
df = df_spark.toPandas()

if "Tc_target" not in df.columns:
    raise ValueError("Table must contain 'Tc_target'.")

y = df["Tc_target"].astype("int32")
groups = build_groups(df)

# COMMAND ----------

leak_cols = {
    "Tc_rul", "Tc_break_date", "Tc_last_fail_if_available_otherwise_null",
    "Tc_r_date", "Wagon_RecordingDate", "Tc_target"
}
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
feat_cols = [c for c in num_cols if c not in leak_cols]
if not feat_cols:
    raise ValueError("No usable numeric feature columns found after excluding leakage columns.")

print(f"[Data] rows={len(df)} | features={len(feat_cols)} | pos_rate~{y.mean():.3f}")

# COMMAND ----------

# MAGIC %md ## Split the data into training and testing and create model
# MAGIC
# MAGIC Create your model here, this is just a reference for you about how to create the model and not limiting your creative ideas.
# MAGIC There are various techiques that you can use for creation of the model such as 
# MAGIC - Right Features
# MAGIC - Right Feature engineering
# MAGIC - Right Feature selection
# MAGIC - Right Train test split
# MAGIC - Right ML/AI model
# MAGIC - Correct Model Hyperparameters
# MAGIC
# MAGIC various others.
# MAGIC
# MAGIC **Remember, You can achieve really good score easily but We would really love to see the techniques that you take to achieve that score. Please Also know that We would check some solutions manually and penaltise the solutions if cheated.** 

# COMMAND ----------

time_col = None
for cand in ["Tc_r_date", "Wagon_RecordingDate"]:
    if cand in df.columns:
        time_col = cand
        break

train_idx, test_idx = None, None
split_mode = ""

if time_col is not None:
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            time_col = None  

if time_col is not None:
    group_ids = groups.astype(str).values
    grp_df = pd.DataFrame({"group": group_ids, time_col: df[time_col].values})
    grp_first_time = grp_df.groupby("group")[time_col].min()
    cutoff = grp_first_time.quantile(0.8)

    train_groups = set(grp_first_time[grp_first_time <= cutoff].index)
    test_groups  = set(grp_first_time[grp_first_time >  cutoff].index)

    train_idx = np.where([g in train_groups for g in group_ids])[0]
    test_idx  = np.where([g in test_groups  for g in group_ids])[0]
    split_mode = f"time-based (cutoff={cutoff})"
else:
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df[feat_cols], y, groups=groups.astype(str).values))
    train_idx, test_idx = folds[0][0], folds[0][1]
    split_mode = "GroupKFold (fallback, fold 0 as test)"

X_train, X_test = df.iloc[train_idx][feat_cols].reset_index(drop=True), df.iloc[test_idx][feat_cols].reset_index(drop=True)
y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
groups_train = groups.iloc[train_idx].reset_index(drop=True)

print(f"[Split] mode={split_mode} | X_train={X_train.shape} | X_test={X_test.shape} | "
      f"pos_rate_train={y_train.mean():.3f} pos_rate_test={y_test.mean():.3f}")

# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

inner_gss = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=RAND_SEED)
C_grid = [0.1, 0.25, 0.5, 1.0, 2.0]  

best_C, best_score = None, -np.inf
for C in C_grid:
    fold_scores = []
    for tr, va in inner_gss.split(X_train, y_train, groups=groups_train):
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]

        pipe_cv = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(
                C=C,
                loss="squared_hinge",
                dual=False,           
                class_weight=None,     
                max_iter=20000,
                tol=1e-3,
                random_state=RAND_SEED
            ))
        ])
        pipe_cv.fit(X_tr, y_tr)
        sc = pipe_cv.decision_function(X_va)
        fold_scores.append(average_precision_score(y_va, sc))
    avg_score = np.nanmean(fold_scores)
    if avg_score > best_score:
        best_score, best_C = avg_score, C

pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=best_C,
        loss="squared_hinge",
        dual=False,
        class_weight=None,          
        max_iter=20000,
        tol=1e-3,
        random_state=RAND_SEED
    ))
])
pipe.fit(X_train, y_train)
print(f"[Model] LinearSVC tuned C={best_C} (inner PR-AUC={best_score:.4f})")

# COMMAND ----------

scores = pipe.decision_function(X_test)

try:
    roc = roc_auc_score(y_test, scores)
except Exception:
    roc = float("nan")
try:
    prauc = average_precision_score(y_test, scores)
except Exception:
    prauc = float("nan")
print(f"ROC-AUC={roc:.4f} | PR-AUC={prauc:.4f}")

prec, rec, thr = precision_recall_curve(y_test, scores)
if len(thr) > 0:
    f1s = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    best_t = thr[f1s.argmax()]
else:
    best_t = 0.0
y_pred = (scores >= best_t).astype(int)

acc  = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)
print(f"Best threshold={best_t:.4f} | Acc={acc:.4f} | BAcc={bacc:.4f}")
print(classification_report(y_test, y_pred, digits=4))

# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

import mlflow 
from mlflow.models.signature import infer_signature 
with mlflow.start_run() as run: 
    ## create signature of the model input and output 
    sign=infer_signature(model_input=X_test.reset_index(drop=True),model_output=y_pred) 
    ## store the model using mlflow 
    mlflow.sklearn.log_model(pipe,model_name ,registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}', signature=sign)

# COMMAND ----------

import pandas as pd
df_result=spark.createDataFrame(pd.DataFrame(
    data=[[model_name,1,str({"accuracy":acc}),0]],
    columns=['ModelName','ModelVersion','ModelMetrics','PipelineVersion']
))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook End
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**
# MAGIC
# MAGIC ####Important: 
# MAGIC 1) Ensure that the result is stored in a PySpark dataframe as
# MAGIC   - 'df_result' e.g. df_result = ...  containing Model Name, Model Version, Performance metrics, Pipeline version (for feature log)
# MAGIC 2) Ensure that your models are registered under ml_catalog. Always name your model as `f'{ml_catalog}.{delta_schema_name}.{model_name}'`
# MAGIC
# MAGIC This will result in model and model config Table in your shared ml_catalog for easy sharing and inference across environments.

# COMMAND ----------

# MAGIC %run "/InsightFactory/Helpers/ML Build (Unity Catalog) Exit"

# COMMAND ----------

# MAGIC %md # Testing or Debugging Zone
