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
# MAGIC | 2025-09-20 | Zi Lun Ma | Transformer which experimented creating sequence per context (p_key). Experimentally integrated 1) new features from feature engineering,  2) Group Lasso, 3) feature pruning, 4) label rule adjustment |
# MAGIC | 2025-10-13 | Zi Lun Ma | 1) refined the train/val/test split, 2) refined sequence building, 3) corrected label definition, 4) implemented WeightedRandomSampler for oversampling, 5) safer MI pruning that averages, 6) added src_key_padding_mask to check if the model is learning nonsense patterns due to padded 0s |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

run_insightfactory = False
run_group_lasso = False

if run_insightfactory:
    skip_training_loop = True
    save_files = False
else:
    skip_training_loop = False
    save_files = True

# COMMAND ----------

model_file_name = "best_transformer_fe.pt"

# COMMAND ----------

#%run "/InsightFactory/Helpers/ML Build (Unity Catalog) Entry"

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

import math, random, os
import numpy as np, pandas as pd
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.feature_selection import mutual_info_classif
import gc
import joblib
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# COMMAND ----------

# config
config = {
    "SEQ_LEN": 30,
    "batch_size": 64,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 40,
    "patience": 20,
    "group_lasso_lambda": 1e-6,
    "keep_frac": 0.8,
    "mi_sample_max": 20000,
    "basecode_emb_dim": 16,
    "focal_alpha": 1.5,
    "focal_gamma": 1.25,
    "seed": 42
}

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

## import your training data
df = spark.sql("""
SELECT *
FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.fe_training
""").toPandas()

p_key_col = "p_key"
recordingDate_col = "Wagon_RecordingDate"
label_col = "Tc_target"
r_date_col = "Tc_r_date"
basecode_col = "Tc_BaseCode"
break_date_col = "Tc_break_date"

for c in (p_key_col, recordingDate_col, label_col, r_date_col, basecode_col):
    assert c in df.columns, f"Need {c} column"

# COMMAND ----------

# feature columns
drop_cols = [
    'Tc_last_fail_if_available_otherwise_null',
    basecode_col,
    label_col,
    'Tc_rul',
    break_date_col,
    r_date_col,
    recordingDate_col,
    p_key_col
]
feature_columns = df.drop(columns=drop_cols, errors='ignore').columns.tolist()

# preprocessing
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# order by context and by recording date
df = df.sort_values([p_key_col, recordingDate_col])

# COMMAND ----------

# build sequences and context metadata
SEQ_LEN = config["SEQ_LEN"]
contexts = []
seqs = []
labels = []
ctx_basecode = {}
ctx_r_year = {}
masks = []

for key, g in df.groupby(p_key_col):
    g = g.sort_values(recordingDate_col)
    # ensure recordingDate <= Tc_r_date
    r_date = pd.to_datetime(g[r_date_col].iloc[0])
    g = g[pd.to_datetime(g[recordingDate_col]) <= r_date]

    Xg = g[feature_columns].values.astype(np.float32)
    if Xg.shape[0] == 0:
        continue

    # break date
    if break_date_col in g.columns:
        break_date = pd.to_datetime(g[break_date_col].iloc[0])
        if pd.isna(break_date):
            continue
    else:
        continue

    # Use the last SEQ_LEN timesteps (pad at front)
    if len(Xg) >= SEQ_LEN:
        seq = Xg[-SEQ_LEN:, :]
        mask = np.zeros(SEQ_LEN, dtype=bool)  # no padding
    else:
        pad = np.zeros((SEQ_LEN - len(Xg), Xg.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, Xg])
        mask = np.ones(SEQ_LEN, dtype=bool)   # padded positions = True
    
    if mask.all():
        # if any sequence is fully padded, replace first token as unmasked
        mask[0] = False

    # Label: 1 if break_date is within 30 days after r_date
    if 0 <= (break_date - r_date).days <= 30:
        label = 1
    else:
        label = 0

    contexts.append(key)
    seqs.append(seq)
    labels.append(label)
    masks.append(mask)

    # store metadata: BaseCode, r_year
    bc = g[basecode_col].iloc[0] if basecode_col in g.columns else "UNK"
    ctx_basecode[key] = bc
    ctx_r_year[key] = int(pd.to_datetime(r_date).year)

X_all = np.stack(seqs)   # shape (N, SEQ_LEN, D)
y_all = np.array(labels).astype(int)
mask_all = torch.from_numpy(np.stack(masks))

print("Contexts:", len(contexts))
print("X_all shape:", X_all.shape)
print("Label dist:", Counter(y_all))

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

# -------------------------
# time-based split by Tc_r_date year
# train: all years except last two available years
# val: penultimate year
# test: last year
# fallback to stratified random split if any split empty
# -------------------------
years = sorted(set(ctx_r_year.values()))
if len(years) >= 3:
    last = years[-1]
    penult = years[-2]
    train_ctx = [c for c in contexts if ctx_r_year[c] < penult]
    val_ctx = [c for c in contexts if ctx_r_year[c] == penult]
    test_ctx = [c for c in contexts if ctx_r_year[c] == last]

    # if any split empty, fallback
    if len(train_ctx) == 0 or len(val_ctx) == 0 or len(test_ctx) == 0:
        print("Time splits produced empty partition; falling back to stratified random split.")
        ctx_train, ctx_temp = train_test_split(contexts, test_size=0.2, random_state=config["seed"], stratify=y_all)
        ctx_val, ctx_test = train_test_split(ctx_temp, test_size=0.5, random_state=config["seed"],
                                             stratify=[y_all[contexts.index(c)] for c in ctx_temp])
        train_ctx, val_ctx, test_ctx = ctx_train, ctx_val, ctx_test
else:
    print("Not enough distinct years for time split; doing stratified random split.")
    ctx_train, ctx_temp = train_test_split(contexts, test_size=0.2, random_state=config["seed"], stratify=y_all)
    ctx_val, ctx_test = train_test_split(ctx_temp, test_size=0.5, random_state=config["seed"],
                                         stratify=[y_all[contexts.index(c)] for c in ctx_temp])
    train_ctx, val_ctx, test_ctx = ctx_train, ctx_val, ctx_test

# map to indices
ctx_to_idx = {c: i for i, c in enumerate(contexts)}
train_idx = [ctx_to_idx[c] for c in train_ctx]
val_idx = [ctx_to_idx[c] for c in val_ctx]
test_idx = [ctx_to_idx[c] for c in test_ctx]

X_train = X_all[train_idx]; y_train = y_all[train_idx]
X_val   = X_all[val_idx];   y_val   = y_all[val_idx]
X_test  = X_all[test_idx];  y_test  = y_all[test_idx]

mask_train = mask_all[train_idx]
mask_val = mask_all[val_idx]
mask_test = mask_all[test_idx]

print("Train/Val/Test sizes:", len(train_idx), len(val_idx), len(test_idx))
print("Train label dist:", Counter(y_train), " Val label dist:", Counter(y_val), " Test:", Counter(y_test))
print("Train/Val/Test contexts:", len(train_ctx), len(val_ctx), len(test_ctx))

# COMMAND ----------

# -------------------------
# aggregated MI pruning (last + mean) — faster + temporal-aware
# -------------------------

prev_features = np.load("feature_columns.npy", allow_pickle=True)
if list(prev_features) == list(feature_columns):
    feature_columns_pruned = np.load("feature_columns_pruned.npy", allow_pickle=True)
    keep_idx = np.array([feature_columns.index(col) for col in feature_columns_pruned])
else:
    D = X_train.shape[2]
    n_sample = min(config["mi_sample_max"], X_train.shape[0])
    rnd_idx = np.random.default_rng(config["seed"]).choice(X_train.shape[0], size=n_sample, replace=False)

    # last timestep features + mean across time
    X_last = X_train[rnd_idx, -1, :]           # (n_sample, D)
    X_mean = X_train[rnd_idx].mean(axis=1)    # (n_sample, D)
    X_mi = np.concatenate([X_last, X_mean], axis=1)  # (n_sample, 2*D)

    # compute mutual information for these aggregated features
    mi_all = mutual_info_classif(X_mi, y_train[rnd_idx], discrete_features=False, random_state=config["seed"], n_neighbors=5)
    # mi_all length = 2*D; average per original feature
    mi_last = mi_all[:D]
    mi_mean = mi_all[D:]
    mi_score = (mi_last + mi_mean) / 2.0
    mi_rank_idx = np.argsort(mi_score)[::-1]

    keep_frac = config["keep_frac"]
    k_keep = max(1, int(np.ceil(D * keep_frac)))
    keep_idx = np.sort(mi_rank_idx[:k_keep])
    feature_columns_pruned = [feature_columns[i] for i in keep_idx]
    if save_files:
        np.save("feature_columns.npy", np.array(feature_columns, dtype=object))
        np.save("feature_columns_pruned.npy", np.array(feature_columns_pruned, dtype=object))
print(f"Pruned features: original {len(feature_columns)} -> kept {len(feature_columns_pruned)}")

# apply pruning
X_train = X_train[:, :, keep_idx]
X_val   = X_val[:, :, keep_idx]
X_test  = X_test[:, :, keep_idx]

# COMMAND ----------

# -------------------------
# scaling: per-BaseCode StandardScaler if enough data, else global
# -------------------------
# Build mapping context -> basecode for train contexts only
train_basecodes = [ctx_basecode[c] for c in train_ctx]
unique_basecodes = sorted(set(train_basecodes))
bc_to_idx = {bc: i for i, bc in enumerate(unique_basecodes)}
idx_to_bc = {i: bc for bc, i in bc_to_idx.items()}

bc_counts = Counter(train_basecodes)

global_scaler = StandardScaler()
global_scaler.fit(X_train.reshape(-1, X_train.shape[2]))

bc_scalers = {}
for bc, count in bc_counts.items():
    idxs = [i for i, c in enumerate(train_ctx) if ctx_basecode[c] == bc]
    X_subset = X_train[idxs].reshape(-1, X_train.shape[2])
    sc = StandardScaler().fit(X_subset)
    bc_scalers[bc] = sc
# apply scaler per-sample based on basecode; fallback to global if need
def scale_3d_per_bc(X3, ctx_list):
    out = np.zeros_like(X3)
    for i, c in enumerate(ctx_list):
        bc = ctx_basecode[c]
        sc = bc_scalers.get(bc, global_scaler)
        out[i] = sc.transform(X3[i].reshape(-1, X3.shape[2])).reshape(X3.shape[1], X3.shape[2])
    return out

# scale data
X_train = scale_3d_per_bc(X_train, train_ctx)
X_val   = scale_3d_per_bc(X_val, val_ctx)
X_test  = scale_3d_per_bc(X_test, test_ctx)


# COMMAND ----------

for bc in unique_basecodes:
    idxs = [i for i, c in enumerate(train_ctx) if ctx_basecode[c] == bc]
    print(bc, Counter(y_train[idxs]))

# COMMAND ----------

if save_files:
    joblib.dump(bc_scalers, "scaler.pkl")
    np.save("bc_to_idx.npy", np.array(bc_to_idx, dtype=object))

# COMMAND ----------


# if save_files:
#     np.save("train_scaled_snapshot.npy", X_train.reshape(-1, X_train.shape[-1]))
#     print("Training scaled stats:",
#         "mean", X_train.mean(),
#         "std", X_train.std(),
#         "min", X_train.min(),
#         "max", X_train.max())

# COMMAND ----------

# -------------------------
# prepare torch datasets and per-BaseCode weighted sampler
# -------------------------
batch_size = config["batch_size"]
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).float()
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val).float()
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test).float()
mask_train_t = mask_train.bool()  # (N_train, SEQ_LEN)
mask_val_t   = mask_val.bool()
mask_test_t  = mask_test.bool()

# Build per-sample weights that incorporate BaseCode imbalance and class imbalance
train_bc_for_idx = [ctx_basecode[c] for c in train_ctx]
bc_class_counts = defaultdict(lambda: Counter())
for bc, label in zip(train_bc_for_idx, y_train):
    bc_class_counts[bc][int(label)] += 1

sample_weights = []
for bc, label in zip(train_bc_for_idx, y_train):
    counts = bc_class_counts[bc]
    w = (sum(counts.values()) + 1) / (counts[int(label)] + 1)
    sample_weights.append(float(w))
sample_weights = np.array(sample_weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


train_dataset = TensorDataset(X_train_t, y_train_t, mask_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t, mask_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t, mask_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# COMMAND ----------

# model with BaseCode embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T]

class TransformerWithBasecode(nn.Module):
    def __init__(self, in_dim, basecode_vocab_size, basecode_emb_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
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
        self.basecode_emb = nn.Embedding(basecode_vocab_size, basecode_emb_dim)
        self.final_ln = nn.LayerNorm(d_model + basecode_emb_dim)
        self.head = nn.Sequential(nn.LayerNorm(d_model + basecode_emb_dim), nn.Linear(d_model + basecode_emb_dim, 1))

    def forward(self, x, basecode_idx, src_mask=None):
        # x: (B, T, F)
        # src_mask: (B, T) bool with True in padded positions
        z = self.input_proj(x)           # (B, T, d_model)
        z = self.pos(z)
        # encoder ignores padded positions when computing attention:
        h = self.encoder(z, src_key_padding_mask=src_mask)   # (B, T, d_model)

        # --- MASKED POOLING (replace simple mean) ---
        if src_mask is None:
            pooled = h.mean(dim=1)  # fallback
        else:
            # src_mask: True = padded -> valid = ~src_mask
            valid = (~src_mask).unsqueeze(-1).to(h.dtype)  # (B, T, 1), float for multiplication
            # zero-out padded positions, sum, divide by valid counts
            summed = (h * valid).sum(dim=1)                # (B, d_model)
            counts = valid.sum(dim=1).clamp(min=1.0)       # (B, 1) avoid div0
            pooled = summed / counts                       # (B, d_model)
        # -----------------------------------------------

        bc_emb = self.basecode_emb(basecode_idx)  # (B, emb)
        concat = torch.cat([pooled, bc_emb], dim=1)
        out = self.head(self.final_ln(concat)).squeeze(-1)
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
basecode_vocab_size = max(1, len(unique_basecodes))
model = TransformerWithBasecode(
    in_dim=X_train.shape[-1],
    basecode_vocab_size=basecode_vocab_size,
    basecode_emb_dim=config["basecode_emb_dim"],
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_layers=config["num_layers"],
    dropout=config["dropout"]
).to(device)

# COMMAND ----------

# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # compute BCEWithLogits per-sample then apply focal term
    def forward(self, logits, targets):
        # logits: (B,), targets: (B,) float 0/1
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)  # p_t = sigmoid(logits) if y=1 else 1-sigmoid(logits)
        focal_term = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_term * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# COMMAND ----------

# Use focal loss as main criterion
criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"], reduction='mean')

# stronger weight decay to discourage memorization
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# COMMAND ----------

# -------------------------
# helper: map context list -> basecode idx tensor
# -------------------------
def ctx_list_to_basecode_idx(ctx_list):
    idxs = []
    for c in ctx_list:
        bc = ctx_basecode.get(c, "UNK")
        idxs.append(bc_to_idx.get(bc, 0))
    return torch.tensor(idxs, dtype=torch.long)

# COMMAND ----------

def log_sampled_distribution(loader, idx_to_bc):
    """
    Inspect actual samples yielded by the train_loader.
    loader: DataLoader that yields (xb, yb, bc_idx)
    idx_to_bc: dict mapping integer basecode index -> basecode string
    """
    global_label_counter = Counter()
    per_bc_counter = {}

    # Go through one full pass
    for xb, yb, bc_idx, mask in loader:
        y_np = yb.cpu().numpy().astype(int)
        bc_np = bc_idx.cpu().numpy()

        for y_val, bc_val in zip(y_np, bc_np):
            global_label_counter[y_val] += 1
            bc_name = idx_to_bc[bc_val]
            if bc_name not in per_bc_counter:
                per_bc_counter[bc_name] = Counter()
            per_bc_counter[bc_name][y_val] += 1

    # Print global distribution
    print("\n===== Sampled label distribution (AFTER sampling) =====")
    for label, count in sorted(global_label_counter.items()):
        print(f"Label {label}: {count}")

    # Print per-basecode distribution
    print("\n===== Sampled per-BaseCode label distribution =====")
    for bc_name, cnt in per_bc_counter.items():
        total = sum(cnt.values())
        print(f"{bc_name}: {dict(cnt)} (total={total})")


# COMMAND ----------

# -------------------------
# evaluation function (pass basecode indices)
# -------------------------
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb, bc_idx, mask in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            bc_idx = bc_idx.to(device)
            mask = mask.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(xb, bc_idx, src_mask=mask)
                probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().astype(int).tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    roc = roc_auc_score(all_labels, all_probs)  if len(set(all_labels)) == 2 else float("nan")
    aupr = average_precision_score(all_labels, all_probs) if len(set(all_labels)) == 2 else float("nan")
    prec, rec, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    return {"acc": acc, "f1": f1, "roc_auc": roc, "aupr": aupr, "prec": prec, "rec": rec}

# COMMAND ----------

def group_lasso_penalty(model, lambda_gl=config["group_lasso_lambda"]):
    """
    Group Lasso penalty on the input projection weights.
    Each input feature (column) is a group.
    """
    if not run_group_lasso:
        return 0
    W = model.input_proj.weight  # shape (d_model, in_dim)
    # l2 norm of each column (feature)
    col_norms = torch.sqrt(torch.sum(W**2, dim=0) + 1e-8)
    return lambda_gl * torch.sum(col_norms)

# COMMAND ----------

# training loop (context-aware basecode indices passed via ctx lists)
if not skip_training_loop:
    best_val_aupr = -np.inf
    stale = 0
    num_epochs = config["num_epochs"]
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # prepare ctx lists corresponding to loaders for evaluation convenience
    train_ctx_list = train_ctx
    val_ctx_list = val_ctx
    test_ctx_list = test_ctx

    # Rebuild train loader that yields (xb, yb, idx) so we can align basecode per sample
    # create explicit indices array for train dataset
    train_indices = np.arange(len(train_idx))
    # we want sampling with replacement according to sample_weights
    sampled_idx = None  # we will not pre-sample; use WeightedRandomSampler with replacement but yield indices via custom collate

    # create a TensorDataset that includes basecode idx per sample in the same order as X_train/y_train
    train_basecode_idx = torch.tensor([bc_to_idx.get(ctx_basecode[c], 0) for c in train_ctx], dtype=torch.long)
    train_dataset = TensorDataset(X_train_t, y_train_t, train_basecode_idx, mask_train_t)
    # sampler already built earlier using sample_weights
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # validation/test dataset include basecode idx
    val_basecode_idx = torch.tensor([bc_to_idx.get(ctx_basecode[c], 0) for c in val_ctx], dtype=torch.long)
    val_dataset = TensorDataset(X_val_t, y_val_t, val_basecode_idx, mask_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_basecode_idx = torch.tensor([bc_to_idx.get(ctx_basecode[c], 0) for c in test_ctx], dtype=torch.long)
    test_dataset = TensorDataset(X_test_t, y_test_t, test_basecode_idx, mask_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        if epoch == 0:
            print("\n*** Logging sampled distribution before training... ***")
            log_sampled_distribution(train_loader, idx_to_bc)

            # Rebuild train_loader because sampler is exhausted
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        model.train()
        total_loss = 0.0
        for xb, yb, bc_idx, mask in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            bc_idx = bc_idx.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(xb, bc_idx, src_mask=mask)
                base_loss = criterion(logits, yb)
                gl_penalty = group_lasso_penalty(model)
                loss = base_loss + gl_penalty

            # backward with scaling if AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / (len(train_loader) + 1e-9)

        # validation
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        print(f"Epoch {epoch+1} loss {avg_loss:.4f} "
            f"val_aupr {val_metrics['aupr']:.4f} "
            f"val_f1 {val_metrics['f1']:.4f} "
            f"val_prec {val_metrics['prec']:.4f} "
            f"val_rec {val_metrics['rec']:.4f}")

        # scheduler step on validation AUPR
        if not np.isnan(val_metrics['aupr']):
            scheduler.step(val_metrics['aupr'])

        # early stopping
        if val_metrics['aupr'] > best_val_aupr + 1e-6:
            best_val_aupr = val_metrics['aupr']
            stale = 0
            torch.save(model.state_dict(), model_file_name)
        else:
            stale += 1
            if stale >= config["patience"]:
                print("Early stopping")
                break

        # cleanup per epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

# Baseline: per-basecode failure rate (train)
from collections import defaultdict
bc_counts = defaultdict(lambda: [0,0])
for c, y in zip(train_ctx, y_train):
    bc = ctx_basecode[c]
    bc_counts[bc][y] += 1

bc_rate = {bc: counts[1] / (counts[0] + counts[1]) for bc, counts in bc_counts.items()}

val_preds_prob = np.array([bc_rate.get(ctx_basecode[c], y_train.mean()) for c in val_ctx])
val_preds = (val_preds_prob >= 0.5).astype(int)

from sklearn.metrics import f1_score, average_precision_score
print("Basecode-only baseline:",
      "AUPR", average_precision_score(y_val, val_preds_prob),
      "F1", f1_score(y_val, val_preds))

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

X_train_last = X_train[:, -1, :]
X_val_last = X_val[:, -1, :]

clf = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_train_last, y_train)
p = clf.predict_proba(X_val_last)[:,1]
print("Logistic last-step:",
      "AUPR", average_precision_score(y_val, p),
      "F1", f1_score(y_val, (p>=0.5).astype(int)))

# COMMAND ----------

# Inspect feature importance
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

with torch.no_grad():
    W = model.input_proj.weight.cpu().numpy()  # (d_model, in_dim)
    col_norms = np.sqrt((W**2).sum(axis=0))
    feature_importance = pd.Series(col_norms, index=feature_columns_pruned)
    print(feature_importance.sort_values())

# COMMAND ----------

# evaluation on test (load best weights)
if os.path.exists(model_file_name):
    model.load_state_dict(torch.load(model_file_name, map_location=device))
    # prepare test loader again if needed
    test_dataset = TensorDataset(X_test_t, y_test_t, test_basecode_idx, mask_test_t)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    test_metrics = evaluate(model, test_loader, device, use_amp=False)
    print("Test metrics:", test_metrics)

# COMMAND ----------


# threshold tuning on validation
model.eval()
all_val_probs, all_val_labels = [], []
with torch.no_grad():
    for xb, yb, bc_idx, mask in val_loader:
        xb = xb.to(device)
        mask = mask.to(device)
        logits = model(xb.to(device).float(), bc_idx.to(device), src_mask=mask).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_val_probs.append(probs)
        all_val_labels.append(yb.numpy())
val_probs = np.concatenate(all_val_probs)
val_labels = np.concatenate(all_val_labels)

best_thr, best_f1 = 0.5, 0
for t in np.linspace(0.01, 0.99, 99):
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_thr, best_f1 = t, f1
print(f"Best threshold on validation: {best_thr:.3f}, F1: {best_f1:.4f}")

# re-evaluate on test using best threshold
all_test_probs, all_test_labels = [], []
with torch.no_grad():
    for xb, yb, bc_idx, mask in test_loader:
        mask = mask.to(device)
        logits = model(xb.to(device).float(), bc_idx.to(device), src_mask=mask).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_test_probs.append(probs)
        all_test_labels.append(yb.numpy())
test_probs = np.concatenate(all_test_probs)
test_labels = np.concatenate(all_test_labels)
test_preds = (test_probs >= best_thr).astype(int)
f1 = f1_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds, zero_division=0)
rec = recall_score(test_labels, test_preds, zero_division=0)
print("Re-evaluated Test Metrics with tuned threshold:")
print(f"F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# COMMAND ----------

# probability distribution check
with torch.no_grad():
    model.eval()
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_probs = []
        for xb, yb, bc_idx in loader:
            xb = xb.to(device).float()
            probs = torch.sigmoid(model(xb, bc_idx)).cpu().numpy()
            all_probs.extend(probs.tolist())
        all_probs = np.array(all_probs)
        print(f"{name} probs: min {all_probs.min():.6f}, "
              f"median {np.median(all_probs):.6f}, "
              f"mean {all_probs.mean():.6f}, "
              f"max {all_probs.max():.6f}")

# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature 

X_sig = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
with torch.no_grad():
    sample_input = torch.tensor(X_test[:100]).float()
    preds = torch.sigmoid(model(sample_input)).numpy()


with mlflow.start_run() as run:
    ## create signature of the model input and output
    sign=infer_signature(model_input=X_sig.reset_index(drop=True),model_output=preds)

    ## store the model using mlflow
    mlflow.pytorch.log_model(model, model_name
                             ,registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}',
                             signature=sign)

# COMMAND ----------

################# Update your output data for the model configuration here #################
import pandas as pd
df_result=spark.createDataFrame(pd.DataFrame(
    data=[[model_name,1,str({"accuracy":test_metrics['acc']}),0]],
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
