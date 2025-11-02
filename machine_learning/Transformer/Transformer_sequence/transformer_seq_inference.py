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
# MAGIC | 2025-09-20 | Zi Lun Ma | Inference script to demonstrate how to load trained features and scaler |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks] databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

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

# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# COMMAND ----------

# MAGIC %pip install tqdm

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm.auto import tqdm

# COMMAND ----------

# preprocess_testing_table -> Tc_p_key
# fe_testing -> p_key

p_key_col = "p_key"
recordingDate_col = "Wagon_RecordingDate"
SEQ_LEN = 30
batch_size = 512
pos_weight = 6.929516315460205  # actual training pos_weight

# COMMAND ----------

input_data=spark.sql('''
      select
      * FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.`predictive_maintenance_uofa_2025`.`fe_testing`
    ''').toPandas().fillna(0)

# COMMAND ----------

# Load training artifacts
scaler = joblib.load("scaler.pkl")  # saved during training
feature_columns = np.load("feature_columns.npy", allow_pickle=True).tolist()

# COMMAND ----------

# build sequences

# sort by p_key then by the same recording date used in training
input_data = input_data.sort_values([p_key_col, recordingDate_col])

groups = []
sorted_keys = []

# iterate groups in same order training used
for key, g in input_data.groupby(p_key_col):
    # ensure the group rows are sorted by recordingDate_col (ascending)
    g = g.sort_values(recordingDate_col)

    # extract features in the exact order saved in feature_columns
    Xg = g[feature_columns].to_numpy(dtype=np.float32)

    # pad on left if shorter, else take last SEQ_LEN
    if Xg.shape[0] < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - Xg.shape[0], Xg.shape[1]), dtype=np.float32)
        Xg_padded = np.vstack([pad, Xg])
    else:
        Xg_padded = Xg[-SEQ_LEN:, :]

    groups.append(Xg_padded)
    sorted_keys.append(key)

# stack to 3D array
X_3d = np.stack(groups)   # shape (K, SEQ_LEN, D)
K, L, D = X_3d.shape
print("Built sequences:", K, "shape:", X_3d.shape)

# COMMAND ----------

# Apply training scaler
X_3d_scaled = scaler.transform(X_3d.reshape(-1, D)).astype(np.float32).reshape(K, L, D)
print("Scaled shape:", X_3d_scaled.shape)
print("Scaled stats: mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}".format(
    X_3d_scaled.mean(), X_3d_scaled.std(), X_3d_scaled.min(), X_3d_scaled.max()
))

# COMMAND ----------

# np.save("infer_scaled_snapshot.npy", X_3d_scaled.reshape(-1, X_3d_scaled.shape[-1]))

# print("Inference scaled stats:",
#       "mean", X_3d_scaled.mean(),
#       "std", X_3d_scaled.std(),
#       "min", X_3d_scaled.min(),
#       "max", X_3d_scaled.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load your model
# MAGIC
# MAGIC Refer to mlflow docs online for right model type if run into errors

# COMMAND ----------

import mlflow
model = mlflow.pytorch.load_model(model_uri)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform inference and save the results of the inference with p_key and target

# COMMAND ----------

# with torch.no_grad():
#     logits = model(Xt)
#     probs = torch.sigmoid(logits).view(-1).cpu().numpy()

Xt = torch.from_numpy(X_3d_scaled).float()  # on CPU for batching
all_probs = np.empty((K,), dtype=np.float32)

for start in tqdm(range(0, K, batch_size), desc="Inferring"):
    end = min(start + batch_size, K)
    batch = Xt[start:end].to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    all_probs[start:end] = probs

print("Probs stats: min {:.6f}, max {:.6f}, mean {:.6f}, median {:.6f}".format(
    all_probs.min(), all_probs.max(), all_probs.mean(), np.median(all_probs)
))
print("Value counts of probs buckets:", np.unique((all_probs*100).astype(int), return_counts=True)[1][:5])


# COMMAND ----------

# Threshold
threshold = 1 / (1 + pos_weight)
pred = (all_probs >= threshold).astype(int)

# COMMAND ----------

# Save predictions

prediction = pd.DataFrame({
    "p_key": sorted_keys,
    "target": pred,
    "probability": all_probs,
})

df_result = spark.createDataFrame(prediction)

# COMMAND ----------


print("probs: min {:.6f}, median {:.6f}, mean {:.6f}, max {:.6f}".format(
    all_probs.min(), np.median(all_probs), all_probs.mean(), all_probs.max()
))
# how many would be positive at various thresholds
for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
    print("thr", t, "pos_count", (all_probs>=t).sum(), "frac", (all_probs>=t).mean())

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
# MAGIC select * from tmp_tc_demo.inferenceSubmission
