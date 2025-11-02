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
# MAGIC | 2025-09-04 | Jinchao Yuan | Initial version. |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks] torch torchvision torchaudio
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

import pandas as pd
import numpy as np
import copy
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

from pyspark.sql import functions as F
## import your training data

# 44 fields
full_data = spark.sql("""SELECT p_key, Tc_BaseCode, Tc_BaseCode_Mapped, Tc_SectionBreakStartKM, Tc_break_date, Tc_last_fail_if_available_otherwise_null, Tc_r_date, Tc_rul, Tc_target, Tng_Tonnage, w_row_count, Wagon_Acc1, Wagon_Acc1_RMS, Wagon_Acc2, Wagon_Acc2_RMS, Wagon_Acc3, Wagon_Acc3_RMS, Wagon_Acc4, Wagon_Acc4_RMS, Wagon_BodyRockFrt, Wagon_BodyRockRr, Wagon_BounceFrt, Wagon_BounceRr, Wagon_BrakeCylinder, Wagon_Curvature, Wagon_ICWVehicle, Wagon_IntrainForce, Wagon_LP1, Wagon_LP2, Wagon_LP3, Wagon_LP4, Wagon_Rail_Pro_L, Wagon_Rail_Pro_R, Wagon_RecordingDate, Wagon_SND, Wagon_SND_L, Wagon_SND_R, Wagon_Speed, Wagon_Track_Offset, Wagon_Twist14m, Wagon_Twist2m, Wagon_VACC, Wagon_VACC_L, Wagon_VACC_R FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.rebalanced_tables.random_oversample_preprocess_table""")

# check the amount of records
print("original rows:", full_data.count())
print("p_key counts:", full_data.select("p_key").distinct().count())

# check if the data includes desired columns
full_data.show(5)

# null check
if full_data.count() == 0:
    print("null!")
else:
    # look into p_key distribution
    pkey_distribution = full_data.groupBy("p_key").count().orderBy(F.desc("count"))
    pkey_distribution.show(10)

# 33 features
feature_columns = [
'Tng_Tonnage', 'Wagon_Acc1', 'Wagon_Acc1_RMS', 'Wagon_Acc2', 'Wagon_Acc2_RMS', 'Wagon_Acc3', 'Wagon_Acc3_RMS', 'Wagon_Acc4', 'Wagon_Acc4_RMS', 'Wagon_BodyRockFrt', 'Wagon_BodyRockRr', 'Wagon_BounceFrt', 'Wagon_BounceRr', 'Wagon_BrakeCylinder', 'Wagon_Curvature', 'Wagon_ICWVehicle', 'Wagon_IntrainForce', 'Wagon_LP1', 'Wagon_LP2', 'Wagon_LP3', 'Wagon_LP4', 'Wagon_Rail_Pro_L', 'Wagon_Rail_Pro_R', 'Wagon_SND', 'Wagon_SND_L', 'Wagon_SND_R', 'Wagon_Speed', 'Wagon_Track_Offset', 'Wagon_Twist14m', 'Wagon_Twist2m', 'Wagon_VACC', 'Wagon_VACC_L', 'Wagon_VACC_R'
]

agg_exprs = []
for col in feature_columns:
    agg_exprs.extend([
        F.min(col).alias(f"{col}_min"),
        F.max(col).alias(f"{col}_max"),
        F.mean(col).alias(f"{col}_mean")
    ])

aggregated_features = (
    full_data
    .groupBy("p_key")
    .agg(*agg_exprs)
)

targets = (
    full_data
    .select("p_key", "Tc_target")
    .dropDuplicates(["p_key", "Tc_target"])
)

final_training_data = (
    aggregated_features
    .join(targets, on="p_key", how="inner")
)

final_training_data = final_training_data.withColumnRenamed("Tc_target", "label")
final_training_data = final_training_data.drop("p_key")


row_count = final_training_data.count()

print(f"final_training_data record number: {row_count}")

pandas_df = final_training_data.toPandas()

# Split the data into features (X) and target (y)
X = pandas_df.drop("label", axis=1)
y = pandas_df["label"]

print(f"final dataset shape: {pandas_df.shape}")
print(f"features counts: {len(X.columns)}")
print(f"sample counts: {len(X)}")

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train_scaled).float()
y_train_tensor = torch.from_numpy(y_train.values).long()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_test_tensor = torch.from_numpy(y_test.values).long()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ImprovedDNN(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedDNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output_layer = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  
        
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

input_dim = X_train_scaled.shape[1]
model = ImprovedDNN(input_dim)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# COMMAND ----------

num_epochs = 30
best_f1 = 0
best_model_state = None  # Store the best model state

# Train the model without MLflow tracking
for epoch in range(num_epochs):
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Epoch {epoch+1}/{num_epochs} started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    model.train()  
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  
        outputs = model(batch_X)  
        loss = criterion(outputs, batch_y)  
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()
    
    # Evaluate the model after each epoch
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        
        # Calculate F1 score
        f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
        
        if f1 > best_f1:
            best_f1 = f1
            # Save the best model state (deep copy to avoid being affected by subsequent training)
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best F1 score: {f1:.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, F1: {f1:.4f}, Best F1: {best_f1:.4f}")
    
    scheduler.step(f1)

    # Calculate epoch duration
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Duration: {epoch_duration:.2f}s")

# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

if best_model_state is not None:
    # Load the best model state
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        final_accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        final_f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
        print(f'accuracy of best model: {final_accuracy:.4f}, f1_score of best model:{final_f1:.4f}')
else:
    print("Training completed but no model improved during training.")

# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

# After all epochs are completed, start MLflow run to log the best model
if best_model_state is not None:
    # Load the best model state
    model.load_state_dict(best_model_state)    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Start MLflow run only once at the end
    with mlflow.start_run():
        # Get a sample for signature inference
        with torch.no_grad():
            sample_output = model(X_test_tensor[:1])
        
        # Infer model signature
        signature = infer_signature(X_test_tensor.numpy()[:1], sample_output.numpy())
        
        # Log the best model to MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}',
            signature=signature
        )
        print(f"Best model registered as: {ml_catalog}.{model_schema_name}.{model_name}")
else:
    print("Training completed but no model improved during training.")

# COMMAND ----------

################# Update your output data for the model configuration here #################
import pandas as pd
df_result=spark.createDataFrame(pd.DataFrame(
    data=[[model_name,1,str({"accuracy":final_accuracy,"F1":final_f1}),0]],
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
