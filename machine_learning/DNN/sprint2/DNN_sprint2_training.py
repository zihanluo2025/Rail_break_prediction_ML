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

# %run "/InsightFactory/Helpers/ML Build (Unity Catalog) Entry"

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

## import your training data
full_df = spark.sql("SELECT Tc_BaseCode, Tc_BaseCode_Mapped, Tc_SectionBreakStartKM, Tc_r_date, Tc_target, Wagon_Twist14m, Wagon_BounceFrt, Wagon_BounceRr, Wagon_BodyRockFrt, Wagon_BodyRockRr, Wagon_LP1, Wagon_LP2, Wagon_LP3, Wagon_LP4, Wagon_Speed, Wagon_BrakeCylinder, Wagon_IntrainForce, Wagon_Acc1, Wagon_Acc2, Wagon_Acc3, Wagon_Acc4, Wagon_Twist2m, Wagon_Acc1_RMS, Wagon_Acc2_RMS, Wagon_Acc3_RMS, Wagon_Acc4_RMS, Wagon_Rail_Pro_L, Wagon_Rail_Pro_R, Wagon_SND, Wagon_VACC, Wagon_VACC_L, Wagon_VACC_R, Wagon_Curvature, Wagon_Track_Offset, Wagon_ICWVehicle, Wagon_SND_L, Wagon_SND_R, w_row_count, Tng_Tonnage FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.`predictive_maintenance_uofa_2025`.`preprocess_training_table`")

pandas_df = full_df.toPandas()

feature_columns = [
    'Wagon_Twist14m', 'Wagon_BounceFrt', 'Wagon_BounceRr', 'Wagon_BodyRockFrt', 
    'Wagon_BodyRockRr', 'Wagon_LP1', 'Wagon_LP2', 'Wagon_LP3', 'Wagon_LP4', 
    'Wagon_Speed', 'Wagon_BrakeCylinder', 'Wagon_IntrainForce', 'Wagon_Acc1', 
    'Wagon_Acc2', 'Wagon_Acc3', 'Wagon_Acc4', 'Wagon_Twist2m', 'Wagon_Acc1_RMS', 
    'Wagon_Acc2_RMS', 'Wagon_Acc3_RMS', 'Wagon_Acc4_RMS', 'Wagon_Rail_Pro_L', 
    'Wagon_Rail_Pro_R', 'Wagon_SND', 'Wagon_VACC', 'Wagon_VACC_L', 'Wagon_VACC_R', 
    'Wagon_Curvature', 'Wagon_Track_Offset', 'Wagon_ICWVehicle', 'Wagon_SND_L', 
    'Wagon_SND_R', 'w_row_count', 'Tng_Tonnage'
]

pandas_df['Tc_r_date'] = pd.to_datetime(pandas_df['Tc_r_date'])

pandas_df['p_key'] = (
    pandas_df['Tc_BaseCode'] + '_' +
    pandas_df['Tc_SectionBreakStartKM'].astype(str) + '_20m_' +
    pandas_df['Tc_r_date'].dt.strftime('%Y-%m-%d')
)

aggregated_features = pandas_df.groupby('p_key')[feature_columns].agg(['min', 'max', 'mean'])
aggregated_features.columns = ['_'.join(col).strip() for col in aggregated_features.columns.values]
aggregated_features = aggregated_features.reset_index()

final_train_data = aggregated_features.merge(
    pandas_df[['p_key'] + ['Tc_target'] + ['Tc_BaseCode_Mapped']].drop_duplicates(),
    on=['p_key'],
    how='inner'
)

final_train_data = final_train_data.rename(columns={'Tc_target': 'label'})
final_train_data = final_train_data.drop(['p_key'], axis=1)

# Split the data into features (X) and target (y)
X = final_train_data.drop('label', axis=1)
y = final_train_data['label']

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
y_train_tensor = torch.from_numpy(y_train.values).long()  # 假设是分类问题，使用long类型为索引
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_test_tensor = torch.from_numpy(y_test.values).long()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

input_dim = X_train_scaled.shape[1]
model = SimpleDNN(input_dim)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  
        outputs = model(batch_X)  
        loss = criterion(outputs, batch_y)  
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'accuracy: {accuracy:.4f}')

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
    sign=infer_signature(model_input=X_test.reset_index(drop=True),model_output=predicted)


    ## store the model using mlflow
    mlflow.sklearn.log_model(model,model_name
                             ,registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}',
                             signature=sign)

# COMMAND ----------

################# Update your output data for the model configuration here #################
import pandas as pd
df_result=spark.createDataFrame(pd.DataFrame(
    data=[[model_name,1,str({"accuracy":accuracy}),0]],
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
