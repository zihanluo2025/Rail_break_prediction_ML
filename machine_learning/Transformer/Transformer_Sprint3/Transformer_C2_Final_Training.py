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
# MAGIC | 2025-09-18 | Di	Zhu | Transformer competition 2: add pos weight and tune hyperparameters.|

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

# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# COMMAND ----------

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

## import your training data
df = spark.sql("""
SELECT *
FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.preprocess_training_table
LIMIT 20000
""")
pandas_df = df.toPandas()

# COMMAND ----------

## Define your features and target
# data columns will not be selected as they are not preprocessed
feature_columns = [
  "Wagon_Twist14m", 
  "Wagon_Twist2m", 
  "Wagon_Speed", 
  "Wagon_BrakeCylinder", 
  "Wagon_IntrainForce", 
  "Wagon_Rail_Pro_L", 
  "Wagon_Rail_Pro_R", 
  "Wagon_Acc4_RMS"                        
]
target_column = "Tc_target"

# COMMAND ----------

# Split the data into features (X) and target (y)
df_copy = pandas_df
X = df_copy[feature_columns]
y = df_copy[target_column]

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

# fix seed
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# COMMAND ----------

# convert X and y to be numpy array
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

# COMMAND ----------

# build windows
def sequence_windows (X, y, past, future):
    X_seq, y_seq = [], []
    for i in range(past, len(X) - future):
        input_seq = X[i-past:i]         
        future_seq = y[i:i+future]
        label = 1 if np.any(future_seq==1) else 0
        X_seq.append(input_seq)
        y_seq.append(label)
    return np.array(X_seq), np.array(y_seq)
X_seq, y_seq = sequence_windows(X, y, 30, 30)
print("X_seq shape:", X_seq.shape)
print("y_seq shape:", y_seq.shape)

# COMMAND ----------

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# COMMAND ----------

# standard scaler
scaler = StandardScaler()
def scaler_and_reshape(X_3d):
    N, L, D = X_3d.shape
    X_2d = X_3d.reshape(-1, D)
    X_2d = scaler.transform(X_2d).astype(np.float32)
    return X_2d.reshape(N, L, D)

N, L, D = X_train.shape
scaler.fit(X_train.reshape(-1, D))
X_train = scaler_and_reshape(X_train)
X_val = scaler_and_reshape(X_val)
X_test = scaler_and_reshape(X_test)

# COMMAND ----------

# DataLoader
batch_size = 64

X_train = torch.from_numpy(np.asarray(X_train)).float()
y_train = torch.from_numpy(np.asarray(y_train)).float()  
X_val = torch.from_numpy(np.asarray(X_val)).float()
y_val = torch.from_numpy(np.asarray(y_val)).float()
X_test = torch.from_numpy(np.asarray(X_test)).float()
y_test = torch.from_numpy(np.asarray(y_test)).float()

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val,y_val)
test_dataset = TensorDataset(X_test,y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# COMMAND ----------

# transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  

    def forward(self, x):  
        T = x.size(1)
        return x + self.pe[:T]

class Transformer(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model) 
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,           
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):  
        z = self.input_proj(x)          
        z = self.pos(z)
        h = self.encoder(z)
        out = h.mean(dim=1)            
        logit = self.head(out).squeeze(-1)  
        return logit

# COMMAND ----------

# Calculate the number of positive and negative class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_num = len(y_train[y_train == 1])    
neg_num = len(y_train[y_train == 0])  
pos_weight = torch.tensor([neg_num/pos_num]).to(device) # weight for positive class  
print("pos num:", pos_num)
print("neg num:", neg_num)
print("pos weight",pos_weight)

# COMMAND ----------

def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).view(-1).cpu().numpy()
            probs.append(p)
            labels.append(yb.view(-1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)

# COMMAND ----------

def training_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, patience=3):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_state, best_val_f1 = None, -1.0
    best_thr = 0.5  
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # fix the threshold as 0.5
        y_prob, y_true = predict_proba(model, val_loader, device)
        y_pred = (y_prob >= best_thr).astype(int)
        cur_f1 = f1_score(y_true, y_pred)

        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "val_best_f1": best_val_f1, "val_best_thr": best_thr}

# COMMAND ----------

def build_model(hp):
    return Transformer(
        in_dim= X_train.shape[-1],
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    )

# COMMAND ----------

def tuing_hyperparameters(train_loader, val_loader):
    hps = {
        "d_model":    [128,256],
        "nhead":      [4,8],
        "num_layers": [2,3],
        "dropout":    [0.2],
        "lr":         [1e-4],
        "weight_decay": [3e-4],
        "epochs":[5]
    }
    best_result  = None
    keys = list(hps.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[hps[k] for k in keys])]

    for i, hp in enumerate(combos, 1):
        model = build_model(hp)

        output = training_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=hp["epochs"],
            lr=hp["lr"],
            patience=3,
        )

        cur_result = {
            "hparams": hp,
            "val_best_f1": output["val_best_f1"],
            "val_best_thr": output["val_best_thr"],
            "model": output["model"],
        }

        print(f"HP {i}: hp:{hp}, F1={cur_result['val_best_f1']:.4f} thr={cur_result['val_best_thr']:.4f}")

        if best_result is None or cur_result["val_best_f1"] > best_result["val_best_f1"]:
            best_result = cur_result
    return best_result

# COMMAND ----------

best = tuing_hyperparameters(train_loader, val_loader)
best_model = best['model']
best_thr = best['val_best_thr']
print(f'Best hyperparameters: {best["hparams"]}, Best val f1: {best["val_best_f1"]}, Best threshold: {best_thr}')

# COMMAND ----------

# get accuracy, auc, and f1 score and preds
def evaluation_metrics(model,loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            logit = model(xb)       
            pred = (torch.sigmoid(logit) >= 0.5).int().reshape(-1) 
            preds.append(pred.cpu().numpy())
            labels.append(yb.cpu().numpy())
    preds  = np.concatenate(preds)         
    labels = np.concatenate(labels).astype(int)   
    acc = accuracy_score(labels, preds) 
    auc = roc_auc_score(labels, preds)  
    f1 = f1_score(labels, preds)
    return acc, auc, f1, preds

# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

# Testing score
accuracy, auc, f1, preds = evaluation_metrics(best_model, test_loader, device)
print(f"Test accuracy = {accuracy:.4f}, Test auc = {auc:.4f}, Test f1 = {f1:.4f}")

# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature 

X_sig = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))

with mlflow.start_run() as run:
    ## create signature of the model input and output
    sign=infer_signature(model_input=X_sig.reset_index(drop=True),model_output=preds)

    ## store the model using mlflow
    mlflow.sklearn.log_model(transformer_model, model_name
                             ,registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}',
                             signature=sign)

# COMMAND ----------

# ################# Update your output data for the model configuration here #################
import pandas as pd
df_result=spark.createDataFrame(pd.DataFrame(
    data=[[model_name,1,str({"accuracy":accuracy,"auc":auc,"f1":f1}),0]],
    columns=['ModelName','ModelVersion','ModelMetrics','PipelineVersion']
))
df_result.write.mode("append").saveAsTable(f"`{ml_catalog}`.`{model_schema_name}`.model_config")

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

# %run "/InsightFactory/Helpers/ML Build (Unity Catalog) Exit"

# COMMAND ----------

# MAGIC %md # Testing or Debugging Zone
