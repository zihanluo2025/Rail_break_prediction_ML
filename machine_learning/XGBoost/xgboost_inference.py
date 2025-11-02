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
# MAGIC | 2025-10-14 | Zi Lun Ma | Inference script for XGBoost |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %pip install xgboost

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

import numpy as np
import pandas as pd
import joblib

# COMMAND ----------

p_key_col = "p_key"
recordingDate_col = "Wagon_RecordingDate"
basecode_col = "Tc_BaseCode"
r_date_col = "Tc_r_date"

CONF = {
    # Artifacts from training:
    # preproc_artifacts.pkl: bc_scalers, global_scaler, feature_columns, SEQ_LEN, train_medians,
    #                        (NEW) bc_cnt, bc_pos, global_prior, m_smooth
    # fe_pipeline.pkl:       agg_scaler, best_threshold,
    #                        (NEW) keep_idx_conservative, q_lo, q_hi, per_bc_thresholds (optional)
    "preproc_out": "preproc_artifacts.pkl",
    "fe_pipeline": "fe_pipeline.pkl",
    "last_k": 5,                    # must match training's k in build_agg_padding_aware
    "default_threshold": 0.50,      # used only if best_threshold absent
}

# COMMAND ----------


# -------------------------
# Load artifacts
# -------------------------
print("Loading artifacts...")
preproc = joblib.load(CONF["preproc_out"])
fe      = joblib.load(CONF["fe_pipeline"])

feature_columns = preproc["feature_columns"]
SEQ_LEN         = preproc.get("SEQ_LEN", 15)
train_medians   = np.asarray(preproc["train_medians"], dtype=np.float32)
bc_scalers      = preproc["bc_scalers"]
global_scaler   = preproc["global_scaler"]

# Optional (new) items – fall back safely if your current files don’t have them yet
bc_cnt          = preproc.get("bc_cnt", {})        # dict BaseCode -> count on TRAIN
bc_pos          = preproc.get("bc_pos", {})        # dict BaseCode -> #positives on TRAIN
global_prior    = float(preproc.get("global_prior", 0.1))
m_smooth        = float(preproc.get("m_smooth", 20.0))

agg_scaler      = fe["agg_scaler"]
best_threshold  = float(fe.get("best_threshold", CONF["default_threshold"]))

# Conservative block selection + clipping window learned on TRAIN
keep_idx_consv  = fe.get("keep_idx_conservative", None)   # np.bool_ mask or np.array indices
q_lo            = fe.get("q_lo", None)
q_hi            = fe.get("q_hi", None)

# Optional per-BC thresholds (learned on VAL)
per_bc_thresholds = fe.get("per_bc_thresholds", None)  # dict(BaseCode -> float) or None


# COMMAND ----------

input_data=spark.sql('''
      select
      * FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.feature_selection.preprocess_predict_table
    ''').withColumnRenamed("Tc_p_key", "p_key").toPandas().fillna(0)

# numeric coercion
input_data[feature_columns] = (
    input_data[feature_columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
)
input_data = input_data.sort_values([p_key_col, recordingDate_col])

# COMMAND ----------

# -------------------------
# Build sequences (up to r_date)
# -------------------------
contexts, seqs, masks, ctx_basecode = [], [], [], {}
for key, g in input_data.groupby(p_key_col):
    g = g.sort_values(recordingDate_col)

    # keep only rows up to r_date (same as training)
    r_date = pd.to_datetime(g[r_date_col].iloc[0])
    g = g[pd.to_datetime(g[recordingDate_col]) <= r_date]

    Xg = g[feature_columns].values.astype(np.float32, copy=False)
    if Xg.shape[0] == 0:
        continue

    if len(Xg) >= SEQ_LEN:
        seq  = Xg[-SEQ_LEN:, :]
        mask = np.zeros(SEQ_LEN, dtype=bool)      # False => real
    else:
        pad  = np.zeros((SEQ_LEN - len(Xg), Xg.shape[1]), dtype=np.float32)
        seq  = np.vstack([pad, Xg])
        mask = np.ones(SEQ_LEN, dtype=bool)       # True  => padded
        if mask.all(): mask[0] = False            # rare guard

    contexts.append(key)
    seqs.append(seq)
    masks.append(mask)
    bc = g[basecode_col].iloc[0] if basecode_col in g.columns else "UNK"
    ctx_basecode[key] = bc

if not seqs:
    raise ValueError("No valid sequences found for inference data!")

X_inf   = np.stack(seqs)       # (N, T, D)
mask_inf= np.stack(masks)      # (N, T)
print("Inference data shape:", X_inf.shape)

# COMMAND ----------

# -------------------------
# Impute padded timesteps with TRAIN medians
# -------------------------
def impute_padded(X, m, med_vec):
    X = X.copy()
    for i in range(X.shape[0]):
        if m[i].any():
            X[i, m[i], :] = med_vec
    return X

X_inf = impute_padded(X_inf, mask_inf, train_medians)

# COMMAND ----------

# -------------------------
# Per-BaseCode Robust scaling (scalers fitted on TRAIN)
# -------------------------
def scale_3d(X3, ctx_ids):
    out = np.zeros_like(X3, dtype=np.float32)
    for i, c in enumerate(ctx_ids):
        bc = ctx_basecode.get(c, None)
        sc = bc_scalers.get(bc, global_scaler)     # unseen BaseCode -> global scaler
        out[i] = sc.transform(X3[i].reshape(-1, X3.shape[2])).reshape(X3.shape[1], X3.shape[2])
    return out

X_inf_s = scale_3d(X_inf, contexts)

# COMMAND ----------

# -------------------------
# Padding-aware aggregation (must mirror training)
# -------------------------
def compute_trend(seq):
    T, D = seq.shape
    if T <= 1: return np.zeros(D, dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    X = seq - seq.mean(axis=0)
    denom = (t - t.mean()) @ (t - t.mean())
    if denom == 0: return np.zeros(D, dtype=np.float32)
    return ((t - t.mean()) @ X) / denom

def build_agg_padding_aware(X3, mask3, k=5):
    N, T, D = X3.shape
    feats = []
    for i in range(N):
        real_idx = ~mask3[i]
        Xi = X3[i][real_idx]
        if Xi.shape[0] == 0:
            Xi = np.zeros((1, D), dtype=np.float32)
        kk = min(k, Xi.shape[0])
        lastk = Xi[-kk:].reshape(-1)
        if kk < k:
            lastk = np.pad(lastk, (0, (k - kk) * D), constant_values=0.0)
        last = Xi[-1]
        mean = Xi.mean(axis=0)
        std  = Xi.std(axis=0)
        mn   = Xi.min(axis=0)
        mx   = Xi.max(axis=0)
        tr   = compute_trend(Xi)
        real_frac = Xi.shape[0] / T
        feats.append(np.concatenate([lastk, last, mean, std, mn, mx, tr,
                                     np.array([real_frac, Xi.shape[0]], dtype=np.float32)], axis=0))
    return np.vstack(feats)

X_inf_agg = build_agg_padding_aware(X_inf_s, mask_inf, k=CONF["last_k"])

# COMMAND ----------

# -------------------------
# Conservative block selection (same blocks as training)
# -------------------------
def select_agg_blocks(D, k=5, keep=('lastk','last','mean','trend','meta')):
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
    return np.concatenate([cols[name] for name in keep])

if keep_idx_consv is None:
    # Backward-compatible fallback: rebuild indices deterministically
    D_feat = X_inf_s.shape[2]
    keep_idx_consv = select_agg_blocks(D_feat, k=CONF["last_k"],
                                       keep=('lastk','last','mean','trend','meta'))

X_inf_agg = X_inf_agg[:, keep_idx_consv]

# COMMAND ----------

# -------------------------
# Quantile clipping to TRAIN window
# -------------------------
def clip_to_train(X, lo, hi):
    return np.minimum(np.maximum(X, lo), hi)

if (q_lo is not None) and (q_hi is not None):
    q_lo = np.asarray(q_lo); q_hi = np.asarray(q_hi)
    X_inf_agg = clip_to_train(X_inf_agg, q_lo, q_hi)
else:
    # Safe fallback: no clip
    pass

# COMMAND ----------

# -------------------------
# BaseCode prior feature (train-only estimate with smoothing)
# -------------------------
def bc_prior_array(bcs):
    out = np.zeros(len(bcs), dtype=np.float32)
    for i, bc in enumerate(bcs):
        n = float(bc_cnt.get(bc, 0))
        s = float(bc_pos.get(bc, 0))
        out[i] = (s + m_smooth * global_prior) / max(1.0, (n + m_smooth))
    return out

inf_bcs = np.array([ctx_basecode[c] for c in contexts])
bc_prior_inf = bc_prior_array(inf_bcs)[:, None]
X_inf_agg = np.hstack([X_inf_agg, bc_prior_inf])

# COMMAND ----------

# -------------------------
# Final Robust scaling on aggregated features
# -------------------------
X_inf_n = agg_scaler.transform(X_inf_agg).astype(np.float32, copy=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load your model
# MAGIC
# MAGIC Refer to mlflow docs online for right model type if run into errors

# COMMAND ----------

import mlflow
clf = mlflow.xgboost.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform inference and save the results of the inference with p_key and target

# COMMAND ----------

probs = clf.predict_proba(X_inf_n)[:, 1].astype(np.float32)

# -------------------------
# Thresholding (per-BC if available)
# -------------------------
def threshold_for_ctx(ctx):
    if isinstance(per_bc_thresholds, dict):
        return float(per_bc_thresholds.get(ctx_basecode_map.get(ctx, "UNK"), best_threshold))
    return float(best_threshold)

preds = np.array([int(probs[i] >= threshold_for_ctx(contexts[i])) for i in range(len(probs))], dtype=int)

print("Probs stats: min {:.6f}, mean {:.6f}, median {:.6f}, max {:.6f}".format(
    probs.min(), probs.mean(), float(np.median(probs)), probs.max()
))
print("Percent positive @threshold:", (preds == 1).mean())

# COMMAND ----------

# Save predictions

prediction = pd.DataFrame({
    "p_key": contexts,
    "target": preds,
    "probability": probs,
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
