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
# MAGIC | 2025-09-15 | Di	Zhu | Transformer + fe_training (complement date). |

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

## Define the data and features that you want to use for the test context
### predictive_maintenance.testcontext contains the records on which you need to predict for evaluating your model and be on the Leader Board
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

input_data=spark.sql('''
      select
      * FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.`predictive_maintenance_uofa_2025`.`fe_testing`
    ''').toPandas().fillna(0)

feature_test_columns = ['Wagon_BounceFrt','fe_Wagon_BounceFrt_Risk', 'Wagon_BounceRr','fe_Wagon_BounceRr_Risk',
 'Wagon_BodyRockFrt','fe_Wagon_BodyRockFrt_Risk','Wagon_BodyRockRr','fe_Wagon_BodyRockRr_Risk','Wagon_LP1',
 'Wagon_LP2','Wagon_LP3','Wagon_LP4','fe_Wagon_LP1_Risk','fe_Wagon_LP2_Risk','fe_Wagon_LP3_Risk','fe_Wagon_LP4_Risk','Wagon_Speed','fe_Wagon_Speed_lag1','fe_Wagon_Speed_lag2','fe_Wagon_Speed_lag3','fe_Wagon_Speed_avg7','fe_Wagon_Speed_avg30','fe_Wagon_Speed_Risk','Wagon_Twist2m','Wagon_Twist14m',
 'fe_Wagon_Twist14m_Risk','fe_Wagon_Twist2m_Risk','Wagon_BrakeCylinder','fe_Wagon_BrakeCylinder_Risk',
 'Wagon_IntrainForce','fe_Wagon_IntrainForce_Risk','fe_Wagon_Acc1_Risk','fe_Wagon_Acc2_Risk','fe_Wagon_Acc3_Risk','fe_Wagon_Acc4_Risk','fe_Wagon_Acc1_RMS_Risk','fe_Wagon_Acc2_RMS_Risk',
 'fe_Wagon_Acc3_RMS_Risk','fe_Wagon_Acc4_RMS_Risk','Wagon_Rail_Pro_L','Wagon_Rail_Pro_R','fe_Wagon_Rail_Pro_LR_Diff','fe_Wagon_Rail_Pro_L_Risk','fe_Wagon_Rail_Pro_R_Risk','Wagon_SND','Wagon_SND_L',
 'Wagon_SND_R','fe_Wagon_SND_LR_Diff','fe_Wagon_SND_Risk','fe_Wagon_SND_L_Risk','fe_Wagon_SND_R_Risk',
 'Wagon_VACC','Wagon_VACC_L','Wagon_VACC_R','fe_Wagon_VACC_Risk', 'fe_Wagon_VACC_L_Risk','fe_Wagon_VACC_R_Risk','fe_VACC_LR_Diff','Wagon_Curvature','Wagon_Track_Offset','Wagon_ICWVehicle','Tng_Tonnage']

# COMMAND ----------

def make_windows_for_inference(df, group_key_col,date_col,feature_cols, L, stride):
  
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    X_list, key_list = [], []
    for gk, g in df.sort_values([group_key_col, date_col]).groupby(group_key_col):
        g = g.sort_values(date_col)
       
        g = g.drop_duplicates(subset=[date_col], keep="last")
        full_idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
        g_full = (g.set_index(date_col)
                    .reindex(full_idx).ffill()
                    .reset_index().rename(columns={"index": date_col}))

        anchor_dates = g_full[date_col].iloc[::max(int(stride), 1)].tolist()
       
        for ad in anchor_dates:
            end_pos = g_full[date_col].searchsorted(ad)
            start_pos = end_pos - (L - 1)

            if start_pos < 0:
                need = -start_pos
                first = g_full.iloc[[0]][feature_cols].to_numpy(np.float32)
                pad = np.repeat(first, need, axis=0)
                win = g_full.iloc[:end_pos + 1][feature_cols].to_numpy(np.float32)
                X = np.vstack([pad, win])[-L:, :]
            else:
                X = g_full.iloc[start_pos:end_pos + 1][feature_cols].to_numpy(np.float32)

           
            pkey = f"{gk}_{ad:%Y-%m-%d}"

            X_list.append(X.astype(np.float32))
            key_list.append(pkey)
    X3d = np.stack(X_list, axis=0).astype(np.float32)
    return X3d, key_list

# COMMAND ----------

input_data["Tc_r_date"]  = pd.to_datetime(input_data["p_key"].str.rsplit(pat="_", n=1).str[-1])
input_data["group_key"]  = input_data["p_key"].str.rsplit(pat="_", n=1).str[0]

# COMMAND ----------

X_3d, p_keys = make_windows_for_inference(
    df=input_data,
    group_key_col="group_key",
    date_col="Tc_r_date",
    feature_cols=feature_test_columns,  
    L=30,                  
    stride=7                       
)

# COMMAND ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K, L, D = X_3d.shape
# scaler
scaler = StandardScaler()
X_3d_scaled = scaler.fit_transform(X_3d.reshape(-1,D)).astype(np.float32).reshape(K,L,D)
Xt = torch.from_numpy(X_3d_scaled).to(device).float()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load your model
# MAGIC
# MAGIC Refer to mlflow docs online for right model type if run into errors

# COMMAND ----------

import mlflow
model=mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform inference and save the results of the inference with p_key and target

# COMMAND ----------

model.eval()
batch = 64
probs = np.empty((K,), dtype=np.float32)
with torch.no_grad():
    for s in range(0, K, batch):
        e = min(s+batch, K)
        xb = torch.from_numpy(X_3d[s:e]).to(device).float()
        logit = model(xb)
        probs[s:e] = torch.sigmoid(logit).view(-1).cpu().numpy()
pred = (probs >= 0.5).astype(int)

prediction = pd.DataFrame({
    'p_key': p_keys,
    'target': pred,
    'probability': probs,
})

df_result = spark.createDataFrame(prediction)

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
