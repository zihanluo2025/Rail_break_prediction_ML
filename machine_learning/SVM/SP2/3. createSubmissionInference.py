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
# MAGIC | 2025-09-16 | Tao Xu | Initial version. |

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

import pandas as pd
import numpy as np

input_df = spark.sql("""
  SELECT
    Tc_BaseCode, Tc_BaseCode_Mapped, Tc_SectionBreakStartKM, Tc_r_date,
    Wagon_Twist14m, Wagon_BounceFrt, Wagon_BounceRr, Wagon_BodyRockFrt,
    Wagon_BodyRockRr, Wagon_LP1, Wagon_LP2, Wagon_LP3, Wagon_LP4,
    Wagon_Speed, Wagon_BrakeCylinder, Wagon_IntrainForce, Wagon_Acc1,
    Wagon_Acc2, Wagon_Acc3, Wagon_Acc4, Wagon_Twist2m, Wagon_Acc1_RMS,
    Wagon_Acc2_RMS, Wagon_Acc3_RMS, Wagon_Acc4_RMS, Wagon_Rail_Pro_L,
    Wagon_Rail_Pro_R, Wagon_SND, Wagon_VACC, Wagon_VACC_L, Wagon_VACC_R,
    Wagon_Curvature, Wagon_Track_Offset, Wagon_ICWVehicle, Wagon_SND_L,
    Wagon_SND_R, w_row_count, Tng_Tonnage
  FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.`predictive_maintenance_uofa_2025`.`preprocess_testing_table`
""").toPandas()


feature_columns = [
    'Wagon_Twist14m','Wagon_BounceFrt','Wagon_BounceRr','Wagon_BodyRockFrt',
    'Wagon_BodyRockRr','Wagon_LP1','Wagon_LP2','Wagon_LP3','Wagon_LP4',
    'Wagon_Speed','Wagon_BrakeCylinder','Wagon_IntrainForce','Wagon_Acc1',
    'Wagon_Acc2','Wagon_Acc3','Wagon_Acc4','Wagon_Twist2m','Wagon_Acc1_RMS',
    'Wagon_Acc2_RMS','Wagon_Acc3_RMS','Wagon_Acc4_RMS','Wagon_Rail_Pro_L',
    'Wagon_Rail_Pro_R','Wagon_SND','Wagon_VACC','Wagon_VACC_L','Wagon_VACC_R',
    'Wagon_Curvature','Wagon_Track_Offset','Wagon_ICWVehicle','Wagon_SND_L',
    'Wagon_SND_R','w_row_count','Tng_Tonnage'
]




# COMMAND ----------

# MAGIC %md
# MAGIC ### Load your model
# MAGIC
# MAGIC Refer to mlflow docs online for right model type if run into errors

# COMMAND ----------

import mlflow
model=mlflow.sklearn.load_model(model_uri)

# COMMAND ----------


try:
    expected = list(model.feature_names_in_)
except AttributeError:

    expected = feature_columns


missing = [c for c in expected if c not in input_df.columns]
for c in missing:
    input_df[c] = 0.0


X_infer = (input_df[expected]
           .copy()
           .fillna(0.0)
           .astype("float32"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform inference and save the results of the inference with p_key and target

# COMMAND ----------

y_pred = model.predict(X_infer)

decision_scores = model.decision_function(X_infer)


positive_probs = 1 / (1 + np.exp(-decision_scores))

prediction = pd.DataFrame(
    {
        "p_key": input_df["p_key"],
        "target": y_pred.astype(int),
        "probability": positive_probs
    }
)

df_result = spark.createDataFrame(prediction)

df_inference_data = spark.createDataFrame(
    input_df[["p_key"] + feature_columns]
)


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
