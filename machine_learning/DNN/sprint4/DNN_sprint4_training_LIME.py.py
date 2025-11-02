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
# MAGIC | 2025-10-18 | Tianhua Zhang  | Initial version. |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks] torch torchvision torchaudio optuna lime
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
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pyspark.sql import functions as F
from pyspark.sql.functions import col, isnull, sum
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import tempfile
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, average_precision_score
import optuna
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

# 43 fields
full_data = spark.sql("""SELECT p_key, Tc_BaseCode, Tc_BaseCode_Mapped, Tc_SectionBreakStartKM, Tc_break_date, Tc_r_date, Tc_rul, Tc_target, Tng_Tonnage, w_row_count, Wagon_Acc1, Wagon_Acc1_RMS, Wagon_Acc2, Wagon_Acc2_RMS, Wagon_Acc3, Wagon_Acc3_RMS, Wagon_Acc4, Wagon_Acc4_RMS, Wagon_BodyRockFrt, Wagon_BodyRockRr, Wagon_BounceFrt, Wagon_BounceRr, Wagon_BrakeCylinder, Wagon_Curvature, Wagon_ICWVehicle, Wagon_IntrainForce, Wagon_LP1, Wagon_LP2, Wagon_LP3, Wagon_LP4, Wagon_Rail_Pro_L, Wagon_Rail_Pro_R, Wagon_RecordingDate, Wagon_SND, Wagon_SND_L, Wagon_SND_R, Wagon_Speed, Wagon_Track_Offset, Wagon_Twist14m, Wagon_Twist2m, Wagon_VACC, Wagon_VACC_L, Wagon_VACC_R FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.preprocess_training_table""")

# 33 numerical features
feature_columns = [
'Tng_Tonnage', 'Wagon_Acc1', 'Wagon_Acc1_RMS', 'Wagon_Acc2', 'Wagon_Acc2_RMS', 'Wagon_Acc3', 'Wagon_Acc3_RMS', 'Wagon_Acc4', 'Wagon_Acc4_RMS', 'Wagon_BodyRockFrt', 'Wagon_BodyRockRr', 'Wagon_BounceFrt', 'Wagon_BounceRr', 'Wagon_BrakeCylinder', 'Wagon_Curvature', 'Wagon_ICWVehicle', 'Wagon_IntrainForce', 'Wagon_LP1', 'Wagon_LP2', 'Wagon_LP3', 'Wagon_LP4', 'Wagon_Rail_Pro_L', 'Wagon_Rail_Pro_R', 'Wagon_SND', 'Wagon_SND_L', 'Wagon_SND_R', 'Wagon_Speed', 'Wagon_Track_Offset', 'Wagon_Twist14m', 'Wagon_Twist2m', 'Wagon_VACC', 'Wagon_VACC_L', 'Wagon_VACC_R'
]

# Create aggregation expressions for each feature column (min, max, mean)
agg_exprs = []
for col in feature_columns:
    agg_exprs.extend([
        F.min(col).alias(f"{col}_min"),
        F.max(col).alias(f"{col}_max"),
        F.mean(col).alias(f"{col}_mean")
    ])

# Aggregate features by p_key using the defined aggregation expressions
aggregated_features = (
    full_data
    .groupBy("p_key")
    .agg(*agg_exprs)
)

# Handle null values by replacing them with 0.0 for all columns except p_key
for col in aggregated_features.columns:
    if col != "p_key":
        aggregated_features = aggregated_features.withColumn(
            col, F.coalesce(F.col(col), F.lit(0.0))
        )

# Extract target variable (Tc_target) for each p_key, removing duplicates
targets = (
    full_data
    .select("p_key", "Tc_target")
    .dropDuplicates(["p_key", "Tc_target"])
)

# Join aggregated features with target variables using p_key
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

# Deal with the imbalance dataset
print("Class distribution in training set:")
print(y_train.value_counts())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class weights: {class_weights}")
mlflow.log_param("class_0_count", y_train.value_counts().get(0, 0))
mlflow.log_param("class_1_count", y_train.value_counts().get(1, 0))
mlflow.log_param("class_weights", class_weights.numpy())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train_scaled).float()
y_train_tensor = torch.from_numpy(y_train.values).long() 
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_test_tensor = torch.from_numpy(y_test.values).long()

# COMMAND ----------

def create_model(trial, input_dim):
    # Suggest hyperparameters for the model architecture
    n_units_l1 = trial.suggest_int('n_units_l1', 32, 128) # Units in first hidden layer
    n_units_l2 = trial.suggest_int('n_units_l2', 16, 64) # Units in second hidden layer
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5) # Dropout rate for regularization
    model = nn.Sequential(
        nn.Linear(input_dim, n_units_l1),  # Input layer to first hidden layer
        nn.ReLU(), # ReLU activation function
        nn.Dropout(dropout_rate), # Dropout for regularization
        nn.Linear(n_units_l1, n_units_l2), # First hidden layer to second hidden layer
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(n_units_l2, 2) # Output layer (2 classes for binary classification)
    )
    return model

def objective_imbalanced(trial):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    X_train_tensor_sub = torch.from_numpy(X_train_sub).float()
    y_train_tensor_sub = torch.from_numpy(y_train_sub.values).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val.values).long()
    
    # Suggest hyperparameters for training
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    model = create_model(trial, X_train_scaled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset_sub = TensorDataset(X_train_tensor_sub, y_train_tensor_sub)
    train_loader_sub = DataLoader(train_dataset_sub, batch_size=batch_size, shuffle=True)
    
    # Use weighted CrossEntropyLoss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for epoch in range(10):
        model.train()
        for batch_X, batch_y in train_loader_sub:
            # Zero out gradients from previous iteration
            optimizer.zero_grad()
            # Forward pass: compute predictions
            outputs = model(batch_X)
            # Compute loss
            loss = criterion(outputs, batch_y)
            # Backward pass: compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        # Convert outputs to probabilities using softmax
        probabilities = torch.softmax(val_outputs, dim=1)
        positive_class_probs = probabilities[:, 1].numpy()
        _, predicted = torch.max(val_outputs, 1)
        
        y_true = y_val_tensor.numpy()
        y_pred = predicted.numpy()
        
        auc_pr = average_precision_score(y_true, positive_class_probs)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        combined_score = 0.7 * auc_pr + 0.3 * f1
    
    return combined_score

# Create Optuna study to maximize the combined score
study = optuna.create_study(direction='maximize')
study.optimize(objective_imbalanced, n_trials=10, show_progress_bar=True)

print("best hyperparameter:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"best AUC-PR: {study.best_value:.4f}")

best_params = study.best_params

# COMMAND ----------

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, best_params):
        super(SimpleDNN, self).__init__()
        # Define network layers using the best hyperparameters from Optuna
        self.layer1 = nn.Linear(input_dim, best_params['n_units_l1'])
        self.layer2 = nn.Linear(best_params['n_units_l1'], best_params['n_units_l2'])
        self.output_layer = nn.Linear(best_params['n_units_l2'], 2)  
        # Define activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(best_params['dropout_rate'])
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
# Get the number of input features from the training data
input_dim = X_train_scaled.shape[1]
# Initialize the model with best hyperparameters from Optuna
model = SimpleDNN(input_dim, best_params)
# Define loss function with class weights to handle imbalanced data
criterion = nn.CrossEntropyLoss(weight=class_weights)
# Initialize Adam optimizer with the best learning rate from Optuna
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

# COMMAND ----------

mlflow.end_run()
run = mlflow.start_run()
run_id = run.info.run_id
print(f"Current Run ID: {run_id}")

mlflow.log_param("batch_size", best_params['batch_size'])
mlflow.log_param("learning_rate", best_params['lr'])
mlflow.log_param("hidden_layer1", best_params['n_units_l1'])
mlflow.log_param("hidden_layer2", best_params['n_units_l2'])
mlflow.log_param("dropout_rate", best_params['dropout_rate'])

# COMMAND ----------

num_epochs = 20
train_losses = []

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
    
    epoch_loss = running_loss/len(train_loader)
    train_losses.append(epoch_loss)
    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    
    probabilities = torch.softmax(test_outputs, dim=1)
    positive_class_probs = probabilities[:, 1].numpy()
    
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    f1 = f1_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    auc_pr = average_precision_score(y_test_tensor.numpy(), positive_class_probs)    
    class_report = classification_report(y_test_tensor.numpy(), predicted.numpy())
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1-score: {f1:.4f}')
    print(f'Test AUC-PR: {auc_pr:.4f}')
    print('\nClassification Report:')
    print(class_report)
    
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_auc_pr", auc_pr)

# COMMAND ----------

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch

# Ensure the model is in evaluation mode
model.eval()

# Prepare data
feature_columns = list(X.columns)
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)

# Define prediction function for LIME
def model_predict_proba(x):
    x_tensor = torch.from_numpy(x).float()
    with torch.no_grad():
        output = model(x_tensor)
        probabilities = torch.softmax(output, dim=1)
        return probabilities.numpy()

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_df.values,
    feature_names=feature_columns,
    class_names=['No_Fracture', 'Fracture'],
    mode='classification',
    discretize_continuous=True
)

print("=" * 80)
print("LIME-based Feature Importance Analysis")
print("=" * 80)

# Analyze multiple samples to get global feature importance
sample_size = min(50, len(X_test_df))
feature_importance_scores = {}

# Initialize importance scores for all features
for feature in feature_columns:
    feature_importance_scores[feature] = []

# Analyze samples
for i in range(sample_size):
    if i % 10 == 0:
        print(f"Analyzing sample {i+1}/{sample_size}")
    
    # Get LIME explanation for this sample
    explanation = explainer.explain_instance(
        X_test_df.iloc[i].values,
        model_predict_proba,
        num_features=len(feature_columns)
    )
    
    # Extract feature importance from explanation
    explanation_list = explanation.as_list()
    for feature_name, importance in explanation_list:
        if feature_name in feature_importance_scores:
            feature_importance_scores[feature_name].append(abs(importance))

# Calculate average importance for each feature
feature_importance = {}
for feature, scores in feature_importance_scores.items():
    if scores:
        feature_importance[feature] = np.mean(scores)
    else:
        feature_importance[feature] = 0.0

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'LIME_Importance': list(feature_importance.values())
}).sort_values('LIME_Importance', ascending=False)

print("\nTop 20 Most Important Features (LIME Analysis):")
print(importance_df.head(20).to_string(index=False))

# Identify top features for potential feature selection
top_features = importance_df.head(20)['Feature'].tolist()
print(f"\nTop 20 features identified: {len(top_features)} features")

print("\n" + "=" * 80)
print("LIME-based Individual Prediction Explanations")
print("=" * 80)
all_pred_probs = model_predict_proba(X_test_scaled[:sample_size])
high_risk_indices = np.where(all_pred_probs[:, 1] >= 0.5)[0]

# If not enough high-risk samples, use the ones with highest probability
if len(high_risk_indices) < 3:
    high_risk_indices = np.argsort(all_pred_probs[:, 1])[-3:][::-1]
    print(f"Note: Only {len(np.where(all_pred_probs[:, 1] >= 0.5)[0])} samples with probability >= 0.5 found.")
    print(f"Using top {len(high_risk_indices)} highest probability samples instead.")

# Analyze high-risk samples
for i, sample_idx in enumerate(high_risk_indices[:3]):
    print(f"\n--- LIME Explanation for High-Risk Sample {i+1} ---")
    
    # Get LIME explanation
    explanation = explainer.explain_instance(
        X_test_df.iloc[sample_idx].values,
        model_predict_proba,
        num_features=15  # Show top 15 features
    )
    
    pred_prob = all_pred_probs[sample_idx, 1]
    print(f"Predicted fracture probability: {pred_prob:.4f}")
    
    # Display explanation
    print("\nFeature Contributions (LIME):")
    explanation_list = explanation.as_list()
    for feature_name, contribution in explanation_list:
        direction = "Increases" if contribution > 0 else "Decreases"
        print(f"  {feature_name}: {contribution:.6f} ({direction} fracture risk)")

print("=" * 80)

# Store top features for potential use in model optimization
print(f"\nLIME Analysis completed. Top {len(top_features)} features identified for potential optimization.")


# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

with torch.no_grad():
    sample_output = model(X_test_tensor[:1])

signature = infer_signature(X_test_tensor.numpy()[:1], sample_output.numpy())

mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}',
    signature=signature
)
print(f"Model registered as: {ml_catalog}.{model_schema_name}.{model_name}")

with tempfile.TemporaryDirectory() as tmp_dir:
    scaler_path = os.path.join(tmp_dir, f"scaler_{run_id}.pkl")
    feature_columns_path = os.path.join(tmp_dir, f"feature_columns_{run_id}.pkl")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(X.columns), feature_columns_path)
    
    mlflow.log_artifact(scaler_path, "preprocessor")
    mlflow.log_artifact(feature_columns_path, "preprocessor")
    
print("Preprocessor artifacts logged to MLflow.")
print(f"Scaler saved to: {scaler_path}")
print(f"Feature columns saved to: {feature_columns_path}")

mlflow.log_param("scaler_path", scaler_path)
mlflow.log_param("feature_columns_path", feature_columns_path)


# COMMAND ----------

print(f"Training completed successfully! Run ID: {run_id}")
mlflow.end_run()

# COMMAND ----------

################# Update your output data for the model configuration here #################
import pandas as pd
df_result=spark.createDataFrame(pd.DataFrame(
    data=[[model_name,1,str({"accuracy":accuracy,"f1":f1,"auc_pr":auc_pr}),0]],
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

# COMMAND ----------

