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
# MAGIC | 2025-10-14 | Zi Lun Ma | XGBoost attempt |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

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

import math, random, os, json
import numpy as np, pandas as pd
from collections import Counter
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# COMMAND ----------

# =====================================================
# CONFIG
# =====================================================
conf = {
    "SEQ_LEN": 15,
    "seed": 42,
    "save_files": True,
    "load_model": False,
    "model_out": "best_xgb_model.pkl",
    "preproc_out": "preproc_artifacts.pkl"
}

# XGBoost hyperparameters
hparam = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "colsample_bynode": 0.8,
    "reg_lambda": 10.0,
    "reg_alpha": 5.0,
    "min_child_weight": 5,
    "gamma": 0.3,
    "random_state": conf["seed"],
    "use_label_encoder": False,
    "early_stopping_rounds": 80,
}

np.random.seed(conf["seed"])
random.seed(conf["seed"])

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

# =====================================================
# LOAD TRAIN DATA
# =====================================================
df = spark.sql("""
SELECT *
FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.eddie_test.training_table
""").toPandas()

p_key_col = "p_key"
recordingDate_col = "Wagon_RecordingDate"
label_col = "Tc_target"
r_date_col = "Tc_r_date"
basecode_col = "Tc_BaseCode"
break_date_col = "Tc_break_date"

drop_cols = [
    'Tc_last_fail_if_available_otherwise_null',
    basecode_col,
    label_col,
    'Tc_rul',
    break_date_col,
    r_date_col,
    recordingDate_col,
    p_key_col,
    'Tc_BaseCode_Mapped',
    'Tc_SectionBreakStartKM',
    'w_row_count',
    'Tng_Tonnage'
]

feature_columns = df.drop(columns=drop_cols, errors='ignore').columns.tolist()
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
df = df.sort_values([p_key_col, recordingDate_col])

# COMMAND ----------

# build sequences per context
SEQ_LEN = conf["SEQ_LEN"]
contexts, seqs, labels = [], [], []
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

    # pad at front if needed
    if len(Xg) >= SEQ_LEN:
        seq = Xg[-SEQ_LEN:, :]
        mask = np.zeros(SEQ_LEN, dtype=bool)  # False -> not padded
    else:
        pad = np.zeros((SEQ_LEN - len(Xg), Xg.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, Xg])
        mask = np.ones(SEQ_LEN, dtype=bool)   # True -> padded
        # guard: if fully padded (rare), unmask first token
        if mask.all():
            mask[0] = False

    # label (same 30-day definition)
    label = 1 if 0 <= (break_date - r_date).days <= 30 else 0

    contexts.append(key)
    seqs.append(seq)
    labels.append(label)
    masks.append(mask)

    bc = g[basecode_col].iloc[0] if basecode_col in g.columns else "UNK"
    ctx_basecode[key] = bc
    ctx_r_year[key] = int(pd.to_datetime(r_date).year)

X_all = np.stack(seqs)             # (N, SEQ_LEN, D)
y_all = np.array(labels).astype(int)
mask_all = np.stack(masks).astype(bool)
mask_all.sum(axis=1) != SEQ_LEN
print("Built sequences:", X_all.shape, "Label dist:", Counter(y_all))

print("Total contexts:", len(contexts))
print("X_all shape:", X_all.shape, "Label dist:", Counter(y_all))

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

# =====================================================
# SPLIT TRAIN/VAL/TEST
# =====================================================
years = sorted(set(ctx_r_year.values()))
if len(years) >= 3:
    last, penult = years[-1], years[-2]
    train_ctx = [c for c in contexts if ctx_r_year[c] < penult]
    val_ctx   = [c for c in contexts if ctx_r_year[c] == penult]
    test_ctx  = [c for c in contexts if ctx_r_year[c] == last]
else:
    ctx_train, ctx_temp = train_test_split(contexts, test_size=0.2, random_state=conf["seed"], stratify=y_all)
    ctx_val, ctx_test = train_test_split(ctx_temp, test_size=0.5, random_state=conf["seed"],
                                         stratify=[y_all[contexts.index(c)] for c in ctx_temp])
    train_ctx, val_ctx, test_ctx = ctx_train, ctx_val, ctx_test

ctx_to_idx = {c:i for i,c in enumerate(contexts)}
train_idx = [ctx_to_idx[c] for c in train_ctx]
val_idx   = [ctx_to_idx[c] for c in val_ctx]
test_idx  = [ctx_to_idx[c] for c in test_ctx]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val,   y_val   = X_all[val_idx],   y_all[val_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

# COMMAND ----------

# ---- Impute padded timesteps with train medians (per feature) ----
# Build masks for each split
mask_train = mask_all[train_idx]
mask_val   = mask_all[val_idx]
mask_test  = mask_all[test_idx]

# Compute per-feature medians on TRAIN using only real (non-padded) timesteps
D = X_train.shape[2]
train_medians = np.zeros(D, dtype=np.float32)
for j in range(D):
    # collect all real timesteps for feature j
    vals = X_train[:, :, j][~mask_train]
    train_medians[j] = np.median(vals) if vals.size else 0.0

# Apply to padded rows in each split
def impute_padded(X, m, med):
    X = X.copy()
    for i in range(X.shape[0]):
        if m[i].any():
            X[i, m[i], :] = med
    return X

X_train = impute_padded(X_train, mask_train, train_medians)
X_val   = impute_padded(X_val,   mask_val,   train_medians)
X_test  = impute_padded(X_test,  mask_test,  train_medians)

# COMMAND ----------

# =====================================================
# ROBUST SCALING
# =====================================================
train_basecodes = [ctx_basecode[c] for c in train_ctx]
unique_bcs = sorted(set(train_basecodes))
global_scaler = RobustScaler().fit(X_train.reshape(-1, X_train.shape[2]))
bc_scalers = {}
for bc in unique_bcs:
    idxs = [i for i,c in enumerate(train_ctx) if ctx_basecode[c] == bc]
    X_subset = X_train[idxs].reshape(-1, X_train.shape[2])
    bc_scalers[bc] = RobustScaler().fit(X_subset)

def scale_3d(X3, ctx_list):
    out = np.zeros_like(X3, float)
    for i,c in enumerate(ctx_list):
        sc = bc_scalers.get(ctx_basecode[c], global_scaler)
        out[i] = sc.transform(X3[i].reshape(-1, X3.shape[2])).reshape(X3.shape[1], X3.shape[2])
    return out

if conf["save_files"]:
    joblib.dump({"bc_scalers": bc_scalers,
                 "global_scaler": global_scaler,
                 "feature_columns": feature_columns,
                 "SEQ_LEN": conf["SEQ_LEN"],
                 "train_medians": train_medians.astype(np.float32)
                },
                conf["preproc_out"])

X_train_s = scale_3d(X_train, train_ctx)
X_val_s   = scale_3d(X_val, val_ctx)
X_test_s  = scale_3d(X_test, test_ctx)

# COMMAND ----------

# === Drift scorers ===
from scipy.stats import ks_2samp

def population_stability_index(expected, actual, bins=20):
    """PSI between two 1D arrays (expected=train, actual=test)."""
    eps = 1e-12
    qs = np.linspace(0, 1, bins+1)
    cuts = np.unique(np.quantile(expected, qs))
    e_hist, _ = np.histogram(expected, bins=cuts)
    a_hist, _ = np.histogram(actual,  bins=cuts)
    e_ratio = np.clip(e_hist / max(1, e_hist.sum()), eps, None)
    a_ratio = np.clip(a_hist / max(1, a_hist.sum()), eps, None)
    return np.sum((a_ratio - e_ratio) * np.log(a_ratio / e_ratio))

# COMMAND ----------

# =====================================================
# PADDING-AWARE AGGREGATION (works on real timesteps only)
# =====================================================
def compute_trend(seq):
    T, D = seq.shape
    if T <= 1:
        return np.zeros(D, dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    X = seq - seq.mean(axis=0)
    denom = (t - t.mean()) @ (t - t.mean())
    if denom == 0:
        return np.zeros(D, dtype=np.float32)
    return ((t - t.mean()) @ X) / denom

def build_agg_padding_aware(X3, mask3, k=5):
    """
    X3: (N,T,D) scaled sequences
    mask3: (N,T) True where padded; False where real
    """
    N, T, D = X3.shape
    feats = []
    for i in range(N):
        real_idx = ~mask3[i]
        Xi = X3[i][real_idx]
        if Xi.shape[0] == 0:
            # Completely missing -> safe zeros
            Xi = np.zeros((1, D), dtype=np.float32)

        # last_k over *available* real timesteps only
        kk = min(k, Xi.shape[0])
        lastk = Xi[-kk:].reshape(-1)
        # If fewer than k, pad the *feature vector* (not sequence time) with zeros
        if kk < k:
            lastk = np.pad(lastk, (0, (k - kk) * D), constant_values=0.0)

        last = Xi[-1]
        mean = Xi.mean(axis=0)
        std  = Xi.std(axis=0)
        mn   = Xi.min(axis=0)
        mx   = Xi.max(axis=0)
        tr   = compute_trend(Xi)

        # add sequence meta features
        real_frac = Xi.shape[0] / T
        feats.append(np.concatenate([lastk, last, mean, std, mn, mx, tr,
                                     np.array([real_frac, Xi.shape[0]], dtype=np.float32)], axis=0))
    return np.vstack(feats)

X_train_agg = build_agg_padding_aware(X_train_s, mask_train, k=5)
X_val_agg   = build_agg_padding_aware(X_val_s,   mask_val,   k=5)
X_test_agg  = build_agg_padding_aware(X_test_s,  mask_test,  k=5)
print("Aggregated shapes (padding-aware):", X_train_agg.shape, X_val_agg.shape, X_test_agg.shape)


# COMMAND ----------

use_conservative_agg = True  # try True; flip back if domain AUC worsens

def select_agg_blocks(Xagg, D, k=5, keep=('lastk','last','mean','trend')):
    """
    Reconstruct column indices by how you built Xagg:
    [lastk(k*D), last(D), mean(D), std(D), mn(D), mx(D), trend(D), meta(2)]
    """
    cols = {}
    off = 0
    cols['lastk'] = np.arange(off, off + k*D); off += k*D
    cols['last']  = np.arange(off, off + D);   off += D
    cols['mean']  = np.arange(off, off + D);   off += D
    cols['std']   = np.arange(off, off + D);   off += D
    cols['mn']    = np.arange(off, off + D);   off += D
    cols['mx']    = np.arange(off, off + D);   off += D
    cols['trend'] = np.arange(off, off + D);   off += D
    cols['meta']  = np.arange(off, off + 2);   off += 2
    keep_idx = np.concatenate([cols[name] for name in keep])
    return keep_idx

if use_conservative_agg:
    D_feat = X_train_s.shape[2]; k_used = 5
    keep_idx = select_agg_blocks(X_train_agg, D_feat, k=k_used,
                                 keep=('lastk','last','mean','trend','meta'))
    X_train_agg = X_train_agg[:, keep_idx]
    X_val_agg   = X_val_agg[:,   keep_idx]
    X_test_agg  = X_test_agg[:,  keep_idx]
    print(f"[Agg] conservative selection → {X_train_agg.shape[1]} cols")

# COMMAND ----------

# Quantile clipping to the train range

q_lo = np.quantile(X_train_agg, 0.005, axis=0)
q_hi = np.quantile(X_train_agg, 0.995, axis=0)

def clip_to_train(X):
    return np.minimum(np.maximum(X, q_lo), q_hi)

X_train_agg = clip_to_train(X_train_agg)
X_val_agg   = clip_to_train(X_val_agg)
X_test_agg  = clip_to_train(X_test_agg)

# COMMAND ----------

# =====================================================
# CORAL (CORrelation ALignment) - align TRAIN to VAL distribution
# =====================================================
use_coral = False  # start with False; turn on if subset+pruning aren't enough

def coral_fit_transform(X_src, X_tgt):
    """Return A,b so that x' = (x - mu_s) @ A + mu_t aligns src to tgt."""
    mu_s = X_src.mean(axis=0, keepdims=True)
    mu_t = X_tgt.mean(axis=0, keepdims=True)
    # covariances
    Cs = np.cov(X_src - mu_s, rowvar=False)
    Ct = np.cov(X_tgt - mu_t, rowvar=False)
    # add small ridge for stability
    r = 1e-3
    # A = (Cs + rI)^-1/2 (Ct + rI)^(1/2)
    # use eigh on symmetric covariances
    def mat_sqrt_inv(M):
        w, V = np.linalg.eigh(M + r*np.eye(M.shape[0]))
        return V @ np.diag(1.0/np.sqrt(np.clip(w, 1e-9, None))) @ V.T
    def mat_sqrt(M):
        w, V = np.linalg.eigh(M + r*np.eye(M.shape[0]))
        return V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    A = mat_sqrt_inv(Cs) @ mat_sqrt(Ct)
    return mu_s, mu_t, A

def coral_apply(X, mu_s, mu_t, A):
    return (X - mu_s) @ A + mu_t

if use_coral:
    mu_s, mu_t, A = coral_fit_transform(X_train_agg, X_val_agg)
    X_train_agg = coral_apply(X_train_agg, mu_s, mu_t, A)

# COMMAND ----------

# =====================================================
# FEATURE DRIFT PRUNING (KS / PSI)
# =====================================================
enable_drift_pruning = False
ks_threshold  = 0.40    # aggressive (0.2 mild, 0.3 medium, 0.45 strong)
psi_threshold = 0.22    # common stability cutoff

if enable_drift_pruning:
    ks_scores  = []
    psi_scores = []
    for j in range(X_train_agg.shape[1]):
        ks = ks_2samp(X_train_agg[:, j], X_val_agg[:, j]).statistic
        psi = population_stability_index(X_train_agg[:, j], X_val_agg[:, j], bins=20)
        ks_scores.append(ks); psi_scores.append(psi)

    ks_scores  = np.array(ks_scores)
    psi_scores = np.array(psi_scores)

    # keep features that are not badly drifted on both metrics
    keep_mask = ~((ks_scores >= ks_threshold) | (psi_scores >= psi_threshold))
    n_drop = int((~keep_mask).sum())
    print(f"[Drift] Dropping {n_drop} / {len(keep_mask)} features (KS≥{ks_threshold} or PSI≥{psi_threshold}).")

    X_train_agg = X_train_agg[:, keep_mask]
    X_val_agg   = X_val_agg[:,   keep_mask]
    X_test_agg  = X_test_agg[:,  keep_mask]
else:
    keep_mask = np.ones(X_train_agg.shape[1], dtype=bool)

# COMMAND ----------

# =====================================================
# BASECODE TARGET PRIOR (train-only, smoothed)
# =====================================================
# Build mapping ctx -> basecode for each split
train_bcs = np.array([ctx_basecode[c] for c in train_ctx])
val_bcs   = np.array([ctx_basecode[c] for c in val_ctx])
test_bcs  = np.array([ctx_basecode[c] for c in test_ctx])

global_prior = float(y_train.mean())
m_smooth = 20.0  # smoothing strength

# counts per BaseCode on TRAIN
from collections import defaultdict
bc_pos = defaultdict(int)
bc_cnt = defaultdict(int)
for bc, y in zip(train_bcs, y_train):
    bc_pos[bc] += int(y)
    bc_cnt[bc] += 1

def bc_prior_array(bcs):
    out = np.zeros(len(bcs), dtype=np.float32)
    for i, bc in enumerate(bcs):
        n = bc_cnt.get(bc, 0)
        s = bc_pos.get(bc, 0)
        # smoothed prior; unseen BaseCodes fall back to global
        out[i] = (s + m_smooth * global_prior) / max(1.0, (n + m_smooth))
    return out

bc_prior_train = bc_prior_array(train_bcs)[:, None]
bc_prior_val   = bc_prior_array(val_bcs)[:, None]
bc_prior_test  = bc_prior_array(test_bcs)[:, None]

# append to aggregated features
X_train_agg = np.hstack([X_train_agg, bc_prior_train])
X_val_agg   = np.hstack([X_val_agg,   bc_prior_val])
X_test_agg  = np.hstack([X_test_agg,  bc_prior_test])


# COMMAND ----------

# ========= Lean pipeline: normalize aggregated features -> XGBoost (class-weight) =========
# 1) Normalize aggregated features (fit on TRAIN; apply to VAL/TEST)
agg_scaler = RobustScaler()
X_train_n = agg_scaler.fit_transform(X_train_agg).astype(np.float32, copy=False)
X_val_n   = agg_scaler.transform(X_val_agg).astype(np.float32, copy=False)
X_test_n  = agg_scaler.transform(X_test_agg).astype(np.float32, copy=False)

# ======== Auto-tune RFF gamma by domain AUC ========
def fit_rff_density_ratio(Xtr, Xte, gammas=(0.2,0.3,0.5,0.8,1.2), comps=(512,1024,2048), seed=42):
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    best = None
    for m in comps:
        for g in gammas:
            rff = RBFSampler(gamma=g, n_components=m, random_state=seed)
            Ztr = rff.fit_transform(Xtr); Zte = rff.transform(Xte)
            Xd = np.vstack([Ztr, Zte])
            yd = np.hstack([np.zeros(len(Ztr),dtype=int), np.ones(len(Zte),dtype=int)])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(Xd, yd)
            auc = roc_auc_score(yd, clf.predict_proba(Xd)[:,1])
            print(f"[RFF] m={m} gamma={g} domain AUC={auc:.3f}")
            if (best is None) or (auc < best[0]): best = (auc,g,m,rff,clf,Ztr,Zte,yd,Xd)
    auc,g,m,rff,clf,Ztr,Zte,yd,Xd = best
    print(f"[RFF] picked m={m}, gamma={g} (domain AUC={auc:.3f})")
    return (auc,g,m,rff,clf,Ztr,Zte,yd,Xd)

# =====================================================
# NON-LINEAR DENSITY-RATIO WEIGHTING (RFF + Logistic)
# =====================================================
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression

use_rff_density_ratio = True
if use_rff_density_ratio:
    domain_auc,g_best,m_best,rff,dom_clf,X_rff_train,X_rff_test,y_dom,X_dom = fit_rff_density_ratio(
        X_train_n, X_test_n, seed=conf["seed"]
    )
    p_test_given_x_train = dom_clf.predict_proba(X_rff_train)[:,1]
    eps = 1e-6
    w_train = p_test_given_x_train / np.clip(1.0 - p_test_given_x_train, eps, None)
    w_train = np.clip(w_train, 0.2, 5.0)
    print(f"[RFF Density Ratio] Domain AUC={domain_auc:.3f} (goal: ↓ toward 0.5)")
else:
    # =====================================================
    # DOMAIN-CLASSIFIER IMPORTANCE WEIGHTING
    # =====================================================
    # Train a small classifier to separate TRAIN vs TEST
    X_dom = np.vstack([X_train_n, X_test_n])
    y_dom = np.hstack([np.zeros(len(X_train_n), dtype=int),
                    np.ones(len(X_test_n),  dtype=int)])

    dom_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        tree_method='hist', random_state=conf["seed"], eval_metric='auc'
    ).fit(X_dom, y_dom)

    p_test_given_x_train = dom_clf.predict_proba(X_train_n)[:, 1]
    domain_auc = roc_auc_score(y_dom, dom_clf.predict_proba(X_dom)[:,1])
    print(f"[Domain] Train-vs-Test ROC AUC (higher = more shift): {domain_auc:.3f}")

    # Convert to importance weights: w(x) = p(test|x) / (1 - p(test|x))
    eps = 1e-6
    w_train = p_test_given_x_train / np.clip(1.0 - p_test_given_x_train, eps, None)
    w_train = np.clip(w_train, 0.2, 5.0)  # avoid extreme variance


# COMMAND ----------

# ---- Test-anchored inlier weight (IsolationForest) ----
from sklearn.ensemble import IsolationForest

# Fit on TEST only (unsupervised "what test looks like"), then score TRAIN
iso = IsolationForest(
    n_estimators=300, max_samples='auto', contamination='auto',
    random_state=conf["seed"]
).fit(X_test_n)

# Higher score => more inlier-like; shift to [0,1]
s_train = iso.decision_function(X_train_n)  # ~[-, +], larger is better
s_test  = iso.decision_function(X_test_n)

# Normalize robustly to [0,1]
def minmax01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    a = np.clip(a, lo, hi)
    return (a - lo) / (hi - lo + 1e-9)

w_iso_train = minmax01(s_train)

# Combine with your density-ratio weights (and optional kNN)
w_train = np.clip(w_train * w_iso_train, 0.05, 10.0)
print(f"[IForest] w_iso mean={w_iso_train.mean():.3f}")

# COMMAND ----------

# =====================================================
# kNN PROXIMITY WEIGHT (no hard subset)
# =====================================================
from sklearn.neighbors import NearestNeighbors

use_knn_weight = True
sigma_scale    = 1.0

if use_knn_weight:
    nn = NearestNeighbors(n_neighbors=1).fit(X_test_n)
    dists, _ = nn.kneighbors(X_train_n, n_neighbors=1, return_distance=True)
    dists = dists.ravel()
    sigma = np.median(dists) * sigma_scale + 1e-9
    w_knn = np.exp(-(dists**2) / (2.0 * sigma**2))
else:
    w_knn = np.ones(len(y_train), dtype=float)

# ---- Combine all weights once (RFF * IForest * kNN) ----
w_total = np.clip(w_train * w_iso_train * w_knn, 0.05, 10.0)
X_train_sel = X_train_n
y_train_sel = y_train
w_train_sel = w_total
print(f"[Weights-only] N={len(y_train_sel)}  w.mean={w_train_sel.mean():.3f}")

# 2) Train XGBoost. Use scale_pos_weight for imbalance.
w_pos = float(w_train_sel[y_train_sel == 1].sum())
w_neg = float(w_train_sel[y_train_sel == 0].sum())
spw_sel = max(1.0, w_neg / max(1.0, w_pos))
print(f"[XGB] weighted spw={spw_sel:.2f}")

xgb_params = dict(
    objective='binary:logistic',
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=12,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=18.0,
    reg_alpha=8.0,
    max_delta_step=1,
    gamma=0.2,
    scale_pos_weight=spw_sel,
    random_state=conf["seed"],
    eval_metric=['aucpr', 'auc'],
    tree_method='hist',
)

# COMMAND ----------

if conf["load_model"] and os.path.exists(conf["model_out"]):
    print("Loading saved model for evaluation only...")
    clf = joblib.load(conf["model_out"])
else:
    clf = xgb.XGBClassifier(**xgb_params)
    clf.set_params(early_stopping_rounds=80)
    # ALWAYS use the selected data + combined weights
    clf.fit(
        X_train_sel, y_train_sel,
        sample_weight=w_train_sel,
        eval_set=[(X_val_n, y_val)],
        verbose=50
    )
    if conf["save_files"]:
        joblib.dump(clf, conf["model_out"])

# COMMAND ----------

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

# ========= Threshold by F1 on validation =========
def metrics(y,p,t=0.5):
    pred=(p>=t).astype(int)
    return dict(
        acc=accuracy_score(y,pred),
        f1=f1_score(y,pred,zero_division=0),
        roc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        aupr=average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,
        prec=precision_score(y,pred,zero_division=0),
        rec=recall_score(y,pred,zero_division=0)
    )

p_train = clf.predict_proba(X_train_sel)[:, 1]  # selected train
p_val   = clf.predict_proba(X_val_n)[:, 1]
p_test  = clf.predict_proba(X_test_n)[:, 1]

# COMMAND ----------

# =====================================================
# PROBABILITY CALIBRATION (isotonic, fit on penultimate year)
# =====================================================
use_calibration = True
if use_calibration:
    cal = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    cal.fit(X_val_n, y_val)
    # overwrite with calibrated probabilities
    p_train = cal.predict_proba(X_train_sel)[:,1]
    p_val   = cal.predict_proba(X_val_n)[:,1]
    p_test  = cal.predict_proba(X_test_n)[:,1]

# ---- Per-BaseCode isotonic calibration (fit on VAL only) ----
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression

scores_val_by_bc = defaultdict(list); labels_val_by_bc = defaultdict(list)
scores_test_by_bc = defaultdict(list); idx_test_by_bc = defaultdict(list)

for i, ctx in enumerate(val_ctx):
    bc = ctx_basecode[ctx]
    scores_val_by_bc[bc].append(p_val[i]); labels_val_by_bc[bc].append(y_val[i])

for i, ctx in enumerate(test_ctx):
    bc = ctx_basecode[ctx]
    scores_test_by_bc[bc].append(p_test[i]); idx_test_by_bc[bc].append(i)

p_test_cal_bc = p_test.copy()
bc_cals = {}
for bc in scores_val_by_bc:
    yv = np.array(labels_val_by_bc[bc]); sv = np.array(scores_val_by_bc[bc])
    if yv.sum() >= 5:  # need a few positives to fit
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(sv, yv)
        bc_cals[bc] = iso
        if bc in scores_test_by_bc:
            ii = idx_test_by_bc[bc]
            p_test_cal_bc[ii] = iso.predict(np.array(scores_test_by_bc[bc]))
print(f"[Per-BC cal] fitted for {len(bc_cals)} basecodes")
# use calibrated per-BC scores hereafter
p_test = p_test_cal_bc

# Apply per-BC calibration to VAL as well (metrics/reporting only)
p_val_cal_bc = p_val.copy()
for bc, iso in bc_cals.items():
    idx = [i for i, ctx in enumerate(val_ctx) if ctx_basecode[ctx] == bc]
    if idx:
        p_val_cal_bc[idx] = iso.predict(p_val_cal_bc[idx])

from sklearn.metrics import precision_recall_curve
target_rec = 0.70
prec, rec, thr = precision_recall_curve(y_val, p_val)
idx = np.where(rec >= target_rec)[0]
best_t = float(thr[max(0, idx[-1]-1)]) if len(idx) else 0.5
print(f"[Thr] Recall-targeted: t={best_t:.3f} (target={target_rec:.2f})")

# COMMAND ----------

from collections import defaultdict
val_scores_by_bc=defaultdict(list); val_labels_by_bc=defaultdict(list)
for s,y,ctx in zip(p_val, y_val, val_ctx):
    bc = ctx_basecode[ctx]; val_scores_by_bc[bc].append(s); val_labels_by_bc[bc].append(y)

def thr_for_bc(scores, labels, target_rec=0.70, min_pos=5):
    scores=np.asarray(scores); labels=np.asarray(labels)
    if labels.sum()<min_pos: return None
    pr, rc, th = precision_recall_curve(labels, scores)
    idx = np.where(rc >= target_rec)[0]
    return float(th[max(0, idx[-1]-1)]) if len(idx) else None

bc_thr={}
for bc in val_scores_by_bc:
    t=thr_for_bc(val_scores_by_bc[bc], val_labels_by_bc[bc], target_rec=0.70)
    if t is not None: bc_thr[bc]=t

def predict_with_bc_threshold(scores, ctx_list, default_t):
    out=np.zeros_like(scores,dtype=int)
    for i,ctx in enumerate(ctx_list):
        out[i]=int(scores[i] >= bc_thr.get(ctx_basecode[ctx], default_t))
    return out

pred_test_bc = predict_with_bc_threshold(p_test, test_ctx, best_t)
print("Test (per-BC):",
      "F1=", f1_score(y_test, pred_test_bc, zero_division=0),
      "Prec=", precision_score(y_test, pred_test_bc, zero_division=0),
      "Rec=", recall_score(y_test, pred_test_bc, zero_division=0))

train_metrics = metrics(y_train_sel, p_train, best_t)
val_metrics   = metrics(y_val,   p_val,   best_t)
test_metrics  = metrics(y_test,  p_test,  best_t)

print("Train metrics:", train_metrics)
print("Val   metrics:", val_metrics)
print("Test  metrics:", test_metrics)

# COMMAND ----------

# =====================================================
# LABEL-PRIOR (CLASS-PRIOR) SHIFT CORRECTION via EM
# =====================================================
def em_prior_shift(p, prior_train, max_iter=50, eps=1e-6):
    """
    p: predicted P(y=1|x) under train prior
    returns: adjusted probabilities under new prior (estimated by EM)
    """
    # initialize with train prior
    pi = np.clip(prior_train, 1e-6, 1-1e-6)
    for _ in range(max_iter):
        # E-step: responsibilities using current prior
        num = p * pi
        den = num + (1 - p) * (1 - pi) + 1e-9
        r = num / den  # expected y=1
        # M-step: update prior from responsibilities
        pi_new = r.mean()
        if abs(pi_new - pi) < eps:
            break
        pi = pi_new
    # Now adjust probabilities to new prior
    num = p * pi
    den = num + (1 - p) * (1 - pi) + 1e-9
    p_adj = num / den
    return np.clip(p_adj, 1e-6, 1 - 1e-6), float(pi)

# train prior (on the selected, weighted train—use unweighted prevalence)
prior_train = float(y_train_sel.mean())

# Adjust TEST probs
use_em_label_shift = False
if use_em_label_shift:
    p_test_adj, pi_hat = em_prior_shift(p_test, prior_train)
    print(f"[Label shift] Estimated test positive prior: {pi_hat:.4f} (train prior: {prior_train:.4f})")

    # Recompute metrics with adjusted probabilities
    test_metrics_adj = metrics(y_test, p_test_adj, best_t)  # or re-search threshold on val if you recalibrate
    print("Test metrics (after label-shift adj):", test_metrics_adj)

# ========= Save artifacts =========

if conf["save_files"]:
    joblib.dump({"agg_scaler": agg_scaler, "best_threshold": best_t}, "fe_pipeline.pkl")

# COMMAND ----------

# =====================================================
# VISUALS: How much shift & how well we adapted
# =====================================================

# 1) Domain-classifier score histograms
plt.figure(figsize=(8,5))
if use_rff_density_ratio:
    p_train_dom = dom_clf.predict_proba(X_rff_train)[:,1]
    p_test_dom  = dom_clf.predict_proba(X_rff_test)[:,1]
else:
    p_train_dom = dom_clf.predict_proba(X_train_n)[:,1]
    p_test_dom  = dom_clf.predict_proba(X_test_n)[:,1]
plt.hist(p_train_dom, bins=40, alpha=0.6, density=True, label='train p(test|x)')
plt.hist(p_test_dom,  bins=40, alpha=0.6, density=True, label='test p(test|x)')
plt.title("Domain classifier post-preprocessing (lower overlap = more shift)")
plt.legend(); plt.xlabel("p(test|x)"); plt.ylabel("density")
plt.tight_layout()
if conf["save_files"]:
    plt.savefig("domain_score_hist.png", dpi=150)
plt.close()

# 2) Domain ROC curve
fpr, tpr, _ = roc_curve(y_dom, dom_clf.predict_proba(X_dom)[:,1])
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f'Domain AUC={domain_auc:.3f}')
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Train vs Test separability')
plt.legend()
plt.tight_layout()
if conf["save_files"]:
    plt.savefig("domain_roc.png", dpi=150)
plt.close()

# 3) (Optional) PCA of aggregated features after mitigation (train vs test)
try:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=conf["seed"])
    min_rows = min(len(X_train_n), len(X_test_n), 3000)
    idx_tr = np.random.choice(len(X_train_n), min_rows, replace=False)
    idx_te = np.random.choice(len(X_test_n),  min_rows, replace=False)
    X_comb = np.vstack([X_train_n[idx_tr], X_test_n[idx_te]])
    X_pca  = pca.fit_transform(X_comb)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:min_rows,0], X_pca[:min_rows,1], s=8, alpha=0.4, label='train')
    plt.scatter(X_pca[min_rows:,0], X_pca[min_rows:,1], s=8, alpha=0.4, label='test')
    plt.title("PCA after padding-aware agg + clipping + BC prior")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout()
    if conf["save_files"]:
        plt.savefig("pca_after_mitigation.png", dpi=150)
    plt.close()
except Exception as e:
    print("PCA plot skipped:", e)

# PCA of the actually used training subset vs test
try:
    from sklearn.decomposition import PCA
    pca2 = PCA(n_components=2, random_state=conf["seed"])
    n_tr2 = min(len(X_train_sel), 3000)
    n_te2 = min(len(X_test_n),   3000)
    idx_tr2 = np.random.choice(len(X_train_sel), n_tr2, replace=False)
    idx_te2 = np.random.choice(len(X_test_n),    n_te2, replace=False)
    X_comb2 = np.vstack([X_train_sel[idx_tr2], X_test_n[idx_te2]])
    X_pca2  = pca2.fit_transform(X_comb2)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca2[:n_tr2,0], X_pca2[:n_tr2,1], s=8, alpha=0.4, label='train_sel')
    plt.scatter(X_pca2[n_tr2:,0], X_pca2[n_tr2:,1], s=8, alpha=0.4, label='test')
    plt.title("PCA of selected train (after weighting/subset) vs test")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout()
    if conf["save_files"]:
        plt.savefig("pca_selected_vs_test.png", dpi=150)
    plt.close()
except Exception as e:
    print("PCA selected plot skipped:", e)


# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

X_sig = pd.DataFrame(X_test_agg)
preds = clf.predict_proba(X_test_agg[:100])[:, 1]

with mlflow.start_run() as run:
    ## create signature of the model input and output
    sign=infer_signature(model_input=X_sig.reset_index(drop=True),model_output=preds)

    ## store the model using mlflow
    mlflow.xgboost.log_model(clf, model_name
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
