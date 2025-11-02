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
# MAGIC | 2025-08-30 | Sheng Wang | Initial version. |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

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

# MAGIC %pip install pandas scikit-learn mlflow-skinny[databricks]
# MAGIC %pip install -U scikit-learn
# MAGIC %pip install -U "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
# MAGIC dbutils.library.restartPython()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC
# MAGIC Here, you define your data and features that will be used to train the model. please use it for your reference and feel free to structure it accordingly.

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


TABLE = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.feature_selection.preprocess_training_table"

def to_date_any(col):
    return F.coalesce(
        F.to_date(col),
        F.to_date(col, "yyyy-MM-dd"),
        F.to_date(col, "yyyy-M-d"),
        F.to_date(col, "yyyy/MM/dd"),
        F.to_date(col, "yyyy/M/d"),
        F.to_date(col, "MM/dd/yyyy"),
        F.to_date(col, "dd/MM/yyyy")
    )


df = spark.table(TABLE).withColumn("Tc_date", to_date_any(F.col("Tc_r_date"))).withColumn("Tc_p_key", F.col("p_key"))

# df_raw = spark.table(TABLE).filter(F.col("Tc_BaseCode") == "ARTC-11")
# df = df_raw.withColumn("Tc_date", to_date_any(F.col("Tc_r_date"))).withColumn("Tc_p_key", F.col("p_key"))

numeric_types = ("double", "float", "int", "bigint", "decimal")
feature_cols = [
    c for c, t in df.dtypes
    if c.startswith("Wagon_")
    and c != "Wagon_RecordingDate"
    and any(tp in t.lower() for tp in numeric_types)
]
print(f"Number of features: {len(feature_cols)}")

df = df.withColumn(
    "sample_id",
    F.concat_ws("_", F.col("Tc_BaseCode"), F.col("Tc_SectionBreakStartKM"), F.col("Tc_date"))
)

for c in feature_cols:
    df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))


agg_exprs = [F.avg(c).alias(c) for c in feature_cols]

# strategy for different agg test
# agg_exprs = []
# for c in feature_cols:
#     agg_exprs.append(F.avg(c).alias(c))

#     # if "Acc" in c:
#         # agg_exprs.append(F.min(c).alias(f"{c}_min"))
#         # agg_exprs.append(F.max(c).alias(f"{c}_max"))
#         # agg_exprs.append(F.stddev(c).alias(f"{c}_std"))


aggregated_df = (
    df.groupBy(
        "Tc_BaseCode",
        "Tc_BaseCode_Mapped",
        "Tc_SectionBreakStartKM",
        "Tc_target",
        "Tc_date",
        "sample_id",
        "Tc_p_key",
    )
    .agg(*agg_exprs)
)

feature_cols = [c for c in aggregated_df.columns if c.startswith("Wagon_")]

sample_cols = ( ["Tc_BaseCode", "Tc_BaseCode_Mapped","Tc_SectionBreakStartKM","Tc_target","Tc_date", "sample_id", "Tc_p_key"] + feature_cols[::] ) 


print(f"Row count after aggregation: {aggregated_df.count():,}")
print(f"Column count after aggregation: {len(aggregated_df.columns)}")

all_features = feature_cols
print(f"all features count: {len(all_features)}")

display(aggregated_df.select(*sample_cols).limit(10))


# COMMAND ----------


import torch.nn.functional as TF

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = TF.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        # α_t: use alpha for positives, and 1 - alpha for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# COMMAND ----------


import torch.nn.functional as TF

class GroupLassoTransformerModel(nn.Module):
    """
    Transformer with Group Lasso regularization
    for feature selection and sequence prediction
    """
    def __init__(self,
                 input_dim,
                 d_model=96,
                 nhead=4,
                 num_layers=1,
                 dim_feedforward=192,
                 dropout=0.1, 
                 max_seq_len=256,
                 feature_groups=None,
                 use_layer_norm=True,
                 use_batch_norm=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.max_seq_len = max_seq_len
        
        # 1) Feature groups
        if feature_groups is None:
            self.feature_groups = [[i] for i in range(input_dim)]
        else:
            self.feature_groups = feature_groups

        # 2) Group projections (dimension-safe)
        if feature_groups is None:
            self.feature_groups = [[i] for i in range(input_dim)]
        else:
            self.feature_groups = feature_groups
        num_groups = len(self.feature_groups)

        # 2) Group projections (dimension-safe)
        self.group_projections = nn.ModuleList()
        self.group_dropouts = nn.ModuleList()
        self.group_sizes = []
        
        group_output_dim = max(1, d_model // max(1, num_groups))  # avoid zero dimension
        for group in self.feature_groups:
            gsz = len(group)
            self.group_sizes.append(gsz)
            self.group_projections.append(nn.Sequential(
                nn.Linear(gsz, group_output_dim),
                nn.LayerNorm(group_output_dim) if use_layer_norm else nn.Identity(),
                nn.GELU(),                       # smoother than ReLU
                nn.Dropout(dropout)
            ))
            self.group_dropouts.append(nn.Dropout(dropout * 0.5))
        
        # Projection dimension alignment
        projected_dim = group_output_dim * len(self.feature_groups)
        if projected_dim != d_model:
            self.dim_adjustment = nn.Sequential(
                nn.Linear(projected_dim, d_model),
                nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout * 0.3)
            )
        else:
            self.dim_adjustment = None

        
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.pos_dropout = nn.Dropout(dropout * 0.3)
        
        # Feature gates - init to 0 (sigmoid -> 0.5)
        self.feature_gates = nn.Parameter(torch.zeros(len(self.feature_groups)))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',  # more stable than gelu here
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Temporal aggregation
        self.temporal_attention = nn.Linear(d_model, 1)
        
        # Output head (simplified)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # More conservative initialization
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                     # Special handling for the final 1-dim layer
                    if hasattr(m, 'out_features') and m.out_features == 1:
                        nn.init.constant_(m.bias, -2.2)  # log(0.1/0.9)
                    else:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, mask=None, training=True, has_data_idx=None):
        B, T, F_all = x.shape
        # === 0) Input safety net: clean NaN/Inf + clip extremes ===
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("!!!Warning: model input contains NaN/Inf, replacing with 0")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_absmax = x.abs().max()
        if x_absmax > 1e4:
            print(f"!!!Warning: input values too large max={float(x_absmax):.2f}, clipping")
            x = torch.clamp(x, min=-100.0, max=100.0)

        # === 1) Mask handling (do NOT infer from has_data; if None, treat all as valid) ===
        if mask is None:
            mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)  # all valid
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        # === 3) Zero out padded time steps to prevent dirty values from propagating ===
        if mask.any():
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        # === 4) Group projection + gating + group-level dropout (with debug fallback) ===
        group_outputs = []
        for i, (group_indices, projection, dropout) in enumerate(
            zip(self.feature_groups, self.group_projections, self.group_dropouts)
        ):
            group_feat = x[:, :, group_indices]           # (B, T, group_size)
            group_out  = self._project_group_debug(i, group_feat, training)

            # More stable gating (avoid extreme 0/1)
            gate_val   = torch.sigmoid(self.feature_gates[i])
            gate_val   = torch.clamp(gate_val, min=0.01, max=0.99)
            group_out  = group_out * gate_val

            if training:
                group_out = dropout(group_out)

            if torch.isnan(group_out).any() or torch.isinf(group_out).any():
                print(f"!!!!Group {i} output contains NaN/Inf")
                group_out = torch.nan_to_num(group_out, nan=0.0, posinf=0.0, neginf=0.0)

            group_outputs.append(group_out)

        x = torch.cat(group_outputs, dim=-1)              # (B, T, d_model')

         # === 5) Dimension alignment ===
        if self.dim_adjustment is not None:
            x = self.dim_adjustment(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # === 6) Positional encoding ===
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_dropout(self.pos_embedding(pos)).unsqueeze(0)
        x = x + pos_emb

        # Transformer input check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("!!Transformer input contains NaN/Inf, replacing with 0")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # === 7) Transformer (with padding mask) ===
        x = self.transformer(x, src_key_padding_mask=mask)

        # Transformer output check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("!!!Transformer output contains NaN/Inf, falling back to mean pooling")
            # Mean pooling (by valid length)
            if mask.any():
                valid_lens = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
                x = x.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / valid_lens
            else:
                x = x.mean(dim=1)
        else:
            # === 8) Attention pooling (stable softmax + mask handling) ===
            attn_scores = self.temporal_attention(x).squeeze(-1)    # (B, T)

            if mask.any():
                attn_scores = attn_scores.masked_fill(mask, -1e4)   # large negative to mask
            # numerical stability: subtract row max
            attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]

            # softmax
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

            # attention weight anomaly handling
            if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
                print("!!!Attention weights abnormal, falling back to mean pooling")
                if mask.any():
                    valid_lens = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
                    x = x.masked_fill(mask.unsqueeze(-1), 0.0).sum(dim=1) / valid_lens
                else:
                    x = x.mean(dim=1)
            else:
                x = torch.sum(x * attn_weights, dim=1)              # (B, d_model)

        # === 9) Output head ===
        logits = self.output_head(x).squeeze(-1)                    # (B,)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("!!!Final output contains NaN/Inf, returning zeros")
            logits = torch.zeros_like(logits)
        else:
            logits = torch.clamp(logits, min=-10.0, max=10.0)

        return logits


    def _project_group_debug(self, i, group_feat, training):
        """Pass through projection[i] layer-by-layer to locate where NaN/Inf occurs and self-heal."""
        seq = self.group_projections[i]
        out = group_feat

        # Parameter health check: ensure this branch's weights are OK
        for n, p in seq.named_parameters():
            if not torch.isfinite(p).all():
                print(f"!!!group[{i}] parameter anomaly: {n} contains NaN/Inf, fixed in-place")
                with torch.no_grad():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
                    p.data.clamp_(-1e3, 1e3)

        for j, layer in enumerate(seq):
            out = layer(out)
            if not torch.isfinite(out).all():
                print(f"!!!!group[{i}] layer {j} {layer.__class__.__name__} produced NaN/Inf")
                # Key stats to help diagnose extreme inputs
                with torch.no_grad():
                    mn = torch.nan_to_num(out).min().item() if out.numel() else 0.0
                    mx = torch.nan_to_num(out).max().item() if out.numel() else 0.0
                print(f"----output range≈[{mn:.3g}, {mx:.3g}],attempting self-heal to 0 and continue")
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if training and hasattr(seq, "dropout"):  # compatibility placeholder; no change
            pass
        return out


    def compute_regularization_loss(self, lambda_l2=0.01, lambda_l1=0.001):
        """
        Compute regularization loss
        """
        l2_loss = 0.0
        l1_loss = 0.0
        # Group Lasso (L2 norm of each group's weights)
        for projection in self.group_projections:
            for module in projection.modules():
                if isinstance(module, nn.Linear):
                    weight_norm = torch.norm(module.weight, p=2)
                    # avoid overly large regularization
                    l2_loss += torch.clamp(weight_norm, max=10.0)
                    break

        # L1 regularization on gates
        l1_loss = torch.norm(self.feature_gates, p=1)
        total_reg = lambda_l2 * l2_loss + lambda_l1 * l1_loss
        
        # avoid overly large regularization term
        return torch.clamp(total_reg, max=1.0)

    def get_feature_importance(self):
        """
        Get per-feature importance
        """
        importance_scores = {}
        with torch.no_grad():
            for i, group_indices in enumerate(self.feature_groups):
                # Gate value
                gate_value = torch.sigmoid(self.feature_gates[i]).item()
                # Projection weight norm
                group_norm = 0.0
                for module in self.group_projections[i].modules():
                    if isinstance(module, nn.Linear):
                        group_norm = torch.norm(module.weight, p=2).item()
                        break
                # Combined importance
                importance = gate_value * group_norm
                
                for idx in group_indices:
                    importance_scores[idx] = importance
        return importance_scores


# COMMAND ----------

from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from pyspark.sql import functions as F

class FastPandasDataset(Dataset):

    def __init__(self, df, feature_cols, label_col='Tc_target',
                 sample_col='sample_id', seq_col='_seq_position',
                 strict=True,
                 clip_mode='percentile',
                 clip_percentile=(0.5, 99.5),     
                 clip_fixed=(-10.0, 10.0)):        

        meta_cols = {label_col, sample_col, seq_col}

        # 1) Force-remove metadata columns from feature list
        self.feature_cols = [c for c in feature_cols if c not in meta_cols]
        leaked = [c for c in feature_cols if c in meta_cols]
        if leaked:
            print(f"[Warn] Removed metadata columns from feature_cols: {leaked}")

        self.label_col  = label_col
        self.sample_col = sample_col
        self.seq_col    = seq_col

        # 2) Select only needed columns: sample_id + features + label
        #    (sample_id is not fed to the model but used for grouping)
        need = [self.sample_col] + self.feature_cols + [self.label_col]
        df_cols = df.columns
        missing = [c for c in need if c not in df_cols]
        if missing and strict:
            raise KeyError(f"Missing columns: {missing}")
        elif missing:
            print(f"[Warn] Missing columns will be ignored: {missing}")
            need = [c for c in need if c in df_cols]

        pdf = df.select(*need).toPandas() if hasattr(df, 'toPandas') else df[need].copy()
        pdf.sort_values([self.sample_col], inplace=True, kind='mergesort')

        # 3) Group by sample_id → (T, F) sequences; Tc_target must be unique
        X_list, y_list = [], []
        for sid, g in pdf.groupby(self.sample_col, sort=False):
            ys = g[self.label_col].dropna().unique()
            assert len(ys) == 1, f"[{sid}] Label not unique or empty: {ys}"
            x_np = g[self.feature_cols].to_numpy(dtype=np.float32)
            X_list.append(torch.from_numpy(np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)))
            y_list.append(float(ys[0]))

        self.X = X_list
        self.y = torch.tensor(y_list, dtype=torch.float32)

        all_samples = np.stack([x.numpy() for x in self.X])  # (N, T, F)
        N, T, F = all_samples.shape

        # Per-feature standardization
        for i in range(all_samples.shape[-1]):
            feature_data = all_samples[:, :, i]
            mean = np.mean(feature_data)
            std = np.std(feature_data) + 1e-6
            if std > 10 or abs(mean) > 10:   # Heuristic threshold; tunable
                feature_data = (feature_data - mean) / std
                all_samples[:, :, i] = feature_data

        # Clipping
        if clip_mode == 'percentile':
            p_low, p_high = clip_percentile
            for i in range(F):
                feat = all_samples[:, :, i]
                lo, hi = np.percentile(feat, [p_low, p_high])
                # If lo==hi (degenerate), skip to avoid zeroing everything
                if lo < hi:
                    feat = np.clip(feat, lo, hi)
                    all_samples[:, :, i] = feat
        elif clip_mode == 'fixed':
            lo, hi = clip_fixed
            all_samples = np.clip(all_samples, lo, hi)
        elif clip_mode == 'none':
            pass
        else:
            raise ValueError(f"Unknown clip_mode: {clip_mode}")

        # Write back to self.X
        self.X = [torch.from_numpy(arr) for arr in all_samples]

        # 4) Self-check prints (ensure sample_id is not in features)
        print("=== Dataset alignment check ===")
        print("Num features:", len(self.feature_cols))
        print("All features:", self.feature_cols[::])
        overlap = set(self.feature_cols) & meta_cols
        assert not overlap, f"Overlap between features and metadata columns: {overlap}"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]

        if x.dtype != torch.float32:
            x = x.float()
        if y.dtype != torch.float32:
            y = y.float()

        return x, y



# COMMAND ----------

from sklearn.metrics import (
    f1_score, precision_recall_curve, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    classification_report
)
import numpy as np

def calculate_metrics(y_true, y_pred_proba, threshold=None):
    """
    Compute comprehensive evaluation metrics — safe version that handles NaNs
    """
     # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # ========== Handle NaNs and abnormal values ==========
    # Check and handle NaNs in y_true
    if np.any(np.isnan(y_true)):
        nan_count = np.sum(np.isnan(y_true))
        print(f"!!!Warning: y_true contains {nan_count} NaN values; they will be removed")
        valid_mask = ~np.isnan(y_true)
        y_true = y_true[valid_mask]
        y_pred_proba = y_pred_proba[valid_mask]
    
    # Check and handle NaNs in y_pred_proba
    if np.any(np.isnan(y_pred_proba)):
        nan_count = np.sum(np.isnan(y_pred_proba))
        print(f"!!!!Warning: predictions contain {nan_count} NaN values")
        # Replace NaNs with a neutral value 0.5
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)
    
    # Check infinities
    if np.any(np.isinf(y_pred_proba)):
        inf_count = np.sum(np.isinf(y_pred_proba))
        print(f"!!!!Warning: predictions contain {inf_count} infinite values")
        y_pred_proba = np.clip(y_pred_proba, -1e10, 1e10)
    
    # Ensure probabilities are in [0, 1]
    if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
        print(f"!!!!Warning: predicted probabilities outside [0,1]; clipping")
        print(f"  Original range: [{np.min(y_pred_proba):.4f}, {np.max(y_pred_proba):.4f}]")
        y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)
    
    # Check if we still have valid data
    if len(y_true) == 0:
        print("!!!!Error: no valid data to compute metrics")
        return get_default_metrics()
    
    # ========== Find optimal threshold ==========
    if threshold is None:
        try:
            # Ensure we have two classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                print(f"⚠️ Warning: only one class {unique_classes}; using default threshold 0.5")
                threshold = 0.5
            else:
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
                # Safe F1 computation
                with np.errstate(divide='ignore', invalid='ignore'):
                    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
                
                if len(thresholds) > 0:
                    best_idx = np.argmax(f1_scores[:-1])
                    threshold = thresholds[best_idx]
                else:
                    threshold = 0.5
        except Exception as e:
            print(f"!!!!Error while computing optimal threshold: {e}")
            threshold = 0.5
    
    # ========== Binarize predictions ==========
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ========== Confusion matrix ==========
    try:
        cm = confusion_matrix(y_true, y_pred)
        # Handle different shapes
        if cm.shape == (1, 1):
            # Predicted only one class
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        elif cm.shape == (1, 2):
            # Ground truth only has negatives, but predictions have two classes
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = 0, 0
        elif cm.shape == (2, 1):
            # Ground truth has two classes, but predictions only have one class
            tn, fn = cm[0, 0], cm[1, 0]
            fp, tp = 0, 0
        else:
            # Normal 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
    except Exception as e:
        print(f"!!!!!Error while computing confusion matrix: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # ========== Metrics ==========
    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }
    
    # ========== AUCs ==========
    try:
        if len(np.unique(y_true)) > 1 and not np.all(y_pred_proba == y_pred_proba[0]):
            # Two classes and predictions are not constant
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
            metrics['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
        else:
            # Not meaningful to compute AUCs
            metrics['roc_auc'] = 0.5
            metrics['pr_auc'] = float(np.mean(y_true)) if len(y_true) > 0 else 0.5
    except Exception as e:
        print(f"!!!!Error computing AUC: {e}")
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.5
    
    # Ensure no NaNs in outputs
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            print(f"!!!!Metric {key} is NaN; replacing with 0")
            metrics[key] = 0.0 if key not in ['roc_auc', 'pr_auc'] else 0.5
    
    return metrics

def get_default_metrics():
    """
    Return default metrics dict (used when computation is not possible)
    """
    return {
        'threshold': 0.5,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'specificity': 0.0,
        'npv': 0.0,
        'tp': 0,
        'fp': 0,
        'tn': 0,
        'fn': 0,
        'roc_auc': 0.5,
        'pr_auc': 0.5
    }

def print_detailed_metrics(metrics, prefix=""):
    """
    Print detailed evaluation metrics — enhanced
    """
    print(f"{prefix}Metrics:")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1-Score:            {metrics['f1']:.4f}")
    print(f"  Specificity:         {metrics['specificity']:.4f}")
    
    # Safe handling for ROC-AUC and PR-AUC
    if 'roc_auc' in metrics and not (isinstance(metrics['roc_auc'], float) and np.isnan(metrics['roc_auc'])):
        print(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
    
    if 'pr_auc' in metrics and not (isinstance(metrics['pr_auc'], float) and np.isnan(metrics['pr_auc'])):
        print(f"  PR-AUC:              {metrics['pr_auc']:.4f}")
    
    print(f"  Best threshold:             {metrics['threshold']:.4f}")
    print(f"  Confusion matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")

    # Extra diagnostics
    total_samples = metrics['tp'] + metrics['fp'] + metrics['tn'] + metrics['fn']
    if total_samples > 0:
        actual_positive = metrics['tp'] + metrics['fn']
        actual_negative = metrics['tn'] + metrics['fp']
        predicted_positive = metrics['tp'] + metrics['fp']
        predicted_negative = metrics['tn'] + metrics['fn']
        
        print(f"\n  Class distribution:")
        print(f"    Actual positives: {actual_positive} ({actual_positive/total_samples:.1%})")
        print(f"    Actual negatives: {actual_negative} ({actual_negative/total_samples:.1%})")
        print(f"    Predicted positives: {predicted_positive} ({predicted_positive/total_samples:.1%})")

# COMMAND ----------

def create_feature_groups_by_prefix(
    feature_cols, 
    prefix_dict=None,   # explicit grouping dictionary
    n_parts=3           # prefix-based grouping rule if not matched (currently unused)
):
    
    from collections import defaultdict
    
    # Build a name -> index map for O(1) lookup
    name2idx = {name: i for i, name in enumerate(feature_cols)}
    groups = []
    assigned = set()  # indices already assigned to any group

    if prefix_dict:  # Rule 1: use explicit dictionary first
         # Create groups according to the provided dictionary
        for group_name, members in prefix_dict.items():
            idxs = []
            for m in members:
                if m in name2idx:
                    j = name2idx[m]
                    if j not in assigned:
                        idxs.append(j)
                        assigned.add(j)
            if idxs:
                groups.append(idxs)

        # Any features not covered by the dictionary become singleton groups
        for j, name in enumerate(feature_cols):
            if j not in assigned:
                groups.append([j])

    else:  # Rule 2: default to singleton groups
        for j, _name in enumerate(feature_cols):
            groups.append([j])

    # Print grouping info (keep the original style)
    print(f"Created {len(groups)} feature groups:")
    for i, grp in enumerate(groups, 1):
        grp_names = [feature_cols[idx] for idx in grp]
        print(f"  Group {i}: {len(grp)} features, e.g.: {grp_names}")
    
    return groups

# COMMAND ----------

prefix_dict = {
    # Twist
    "Wagon_Twist": ["Wagon_Twist14m", "Wagon_Twist2m"],
    # Bounce
    "Wagon_Bounce": ["Wagon_BounceFrt", "Wagon_BounceRr"],
    # Body rock
    "Wagon_BodyRock": ["Wagon_BodyRockFrt", "Wagon_BodyRockRr"],
     # Longitudinal position (LP series)
    "Wagon_LP": ["Wagon_LP1", "Wagon_LP2", "Wagon_LP3", "Wagon_LP4"],
    # Speed / Brake / Traction
    "Wagon_Speed": ["Wagon_Speed"],
    "Wagon_Brake": ["Wagon_BrakeCylinder"],
    "Wagon_Force": ["Wagon_IntrainForce"],
    # Acceleration
    "Wagon_Acc": [
        "Wagon_Acc1", "Wagon_Acc1_RMS",
        "Wagon_Acc2", "Wagon_Acc2_RMS",
        "Wagon_Acc3", "Wagon_Acc3_RMS",
        "Wagon_Acc4", "Wagon_Acc4_RMS"
    ],
    # Track geometry
    "Wagon_Rail": ["Wagon_Rail_Pro_L", "Wagon_Rail_Pro_R"],
    "Wagon_Curvature": ["Wagon_Curvature"],
    "Wagon_Track": ["Wagon_Track_Offset"],
    # Noise / Vibration
    "Wagon_SND": ["Wagon_SND", "Wagon_SND_L", "Wagon_SND_R"],
    "Wagon_VACC": ["Wagon_VACC", "Wagon_VACC_L", "Wagon_VACC_R"],
    # Vehicle identifier
    "Wagon_ICWVehicle": ["Wagon_ICWVehicle"],
    # Time-related
    "time_features": ["days_to_target", "time_position", "days_since_last_fail"],
    "cyclic_features": ["doy_sin", "doy_cos"],
    #  History / status flags
    "flags": ["has_last_fail", "has_data"],
},

prefix_dict_no_group = {
    # Twist
    "Wagon_Twist14m": ["Wagon_Twist14m"],
    "Wagon_Twist2m":  ["Wagon_Twist2m"],
    # Bounce
    "Wagon_BounceFrt": ["Wagon_BounceFrt"],
    "Wagon_BounceRr":  ["Wagon_BounceRr"],
    # Body rock
    "Wagon_BodyRockFrt": ["Wagon_BodyRockFrt"],
    "Wagon_BodyRockRr":  ["Wagon_BodyRockRr"],
    # Longitudinal position (LP)
    "Wagon_LP1":  ["Wagon_LP1"],
    "Wagon_LP2":  ["Wagon_LP2"],
    "Wagon_LP3":  ["Wagon_LP3"],
    "Wagon_LP4":  ["Wagon_LP4"],
     # Speed / Brake / Traction
    "Wagon_Speed":  ["Wagon_Speed"],
    "Wagon_BrakeCylinder":  ["Wagon_BrakeCylinder"],
    "Wagon_IntrainForce":  ["Wagon_IntrainForce"],
    # Acceleration
    "Wagon_Acc1":  ["Wagon_Acc1"],
    "Wagon_Acc1_RMS":  ["Wagon_Acc1_RMS"],
    "Wagon_Acc2":  ["Wagon_Acc2"],
    "Wagon_Acc2_RMS":  ["Wagon_Acc2_RMS"],
    "Wagon_Acc3":  ["Wagon_Acc3"],
    "Wagon_Acc3_RMS":  ["Wagon_Acc3_RMS"],
    "Wagon_Acc4":  ["Wagon_Acc4"],
    "Wagon_Acc4_RMS":  ["Wagon_Acc4_RMS"],
    # Track geometry
    "Wagon_Rail_Pro_L":  ["Wagon_Rail_Pro_L"],
    "Wagon_Rail_Pro_R":  ["Wagon_Rail_Pro_R"],
    "Wagon_Curvature":  ["Wagon_Curvature"],
    "Wagon_Track_Offset":  ["Wagon_Track_Offset"],
    # Noise / Vibration
    "Wagon_SND":  ["Wagon_SND"],
    "Wagon_SND_L":  ["Wagon_SND_L"],
    "Wagon_SND_R":  ["Wagon_SND_R"],
    "Wagon_VACC":  ["Wagon_VACC"],
    "Wagon_VACC_L":  ["Wagon_VACC_L"],
    "Wagon_VACC_R":  ["Wagon_VACC_R"],

}



prefix_dict_with_group = {
    # Twist
    "Wagon_Twist": ["Wagon_Twist14m","Wagon_Twist2m"],
    # Bounce
    "Wagon_Bounce": ["Wagon_BounceFrt","Wagon_BounceRr"],
    # Body rock
    "Wagon_BodyRock": ["Wagon_BodyRockFrt","Wagon_BodyRockRr"],
    # Longitudinal position (LP)
    "Wagon_LP":  ["Wagon_LP1","Wagon_LP2","Wagon_LP3","Wagon_LP4"],
     # Speed / Brake / Traction
    "Wagon_Speed":  ["Wagon_Speed","Wagon_BrakeCylinder","Wagon_IntrainForce"],

    # Acceleration
    "Wagon_Acc":  [
        "Wagon_Acc1",
        "Wagon_Acc2",
        "Wagon_Acc3",
        "Wagon_Acc4",
        "Wagon_Acc1_RMS",
        "Wagon_Acc2_RMS",
        "Wagon_Acc3_RMS",
        "Wagon_Acc4_RMS",
    ],

    # Track geometry
    "Wagon_Track geometry":  ["Wagon_Rail_Pro_L","Wagon_Rail_Pro_R","Wagon_Curvature","Wagon_Track_Offset"],

    # Noise / Vibration
    "Wagon_SND":  ["Wagon_SND","Wagon_SND_L","Wagon_SND_R"],
    "Wagon_VACC":  ["Wagon_VACC","Wagon_VACC_L","Wagon_VACC_R"]
}


# COMMAND ----------

def filter_feature_groups(selected_features, prefix_dict_with_group):
    new_groups = {}

    for group_name, prefix_list in prefix_dict_with_group.items():
        group_feats = []

        for feat in selected_features:
            if feat in prefix_list:  # excatly mating
                group_feats.append(feat)

        if group_feats:
            new_groups[group_name] = group_feats

    return new_groups

# COMMAND ----------

def train_loop(model, train_loader, valid_loader, 
               optimizer, scheduler, criterion,
               n_epochs=30, patience=10, 
               lambda_l2=1e-4, lambda_l1=1e-5,
               device="cpu"):
    history = {
        'train_loss': [], 'train_metrics': [],
        'valid_loss': [], 'valid_metrics': [],
        'reg_loss': [], 'feature_importance': []
    }
    best_valid_f1, best_model_state, best_metrics = 0, None, None
    best_epoch, patience_counter = 0, 0
    print("\nStart training...")
    print("="*60)

    for epoch in range(n_epochs):
        # ========== Training ==========
        model.train()
        train_losses, train_bce_losses, train_reg_losses = [], [], []
        train_preds, train_labels = [], []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
            optimizer.zero_grad()

            # Forward pass (train)
            outputs = model(batch_x, training=True)
            # Compute loss
            bce_loss = criterion(outputs, batch_y)

            reg_scale = min(1.0, (epoch + 1) / 5)
            reg_loss = reg_scale * model.compute_regularization_loss(lambda_l2, lambda_l1)
            total_loss = bce_loss + reg_loss

            if not torch.isfinite(total_loss):
                print("!!!!total_loss is not finite; skip this batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Backward
            total_loss.backward()
            bad = False

            for n, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print(f"!!!!Gradient NaN/Inf: {n} -> skip this batch")
                    bad = True
                    p.grad = None
            if bad:
                optimizer.zero_grad(set_to_none=True)
                continue
        
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_losses.append(total_loss.item())
            train_bce_losses.append(bce_loss.item())
            train_reg_losses.append(reg_loss.item())
            with torch.no_grad():
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

        train_metrics = calculate_metrics(train_labels, train_preds)
        avg_train_loss = np.mean(train_losses)
        avg_train_bce = np.mean(train_bce_losses)
        avg_train_reg = np.mean(train_reg_losses)

        # ========== Validation ==========
        model.eval()
        valid_losses, valid_bce_losses, valid_reg_losses = [], [], []
        valid_preds, valid_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()

                outputs = model(batch_x, training=False)

                bce_loss = criterion(outputs, batch_y)
                reg_loss = model.compute_regularization_loss(lambda_l2, lambda_l1)
                total_loss = bce_loss + reg_loss

                valid_losses.append(total_loss.item())
                valid_bce_losses.append(bce_loss.item())
                valid_reg_losses.append(reg_loss.item())
                valid_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                valid_labels.extend(batch_y.cpu().numpy())

        valid_metrics = calculate_metrics(valid_labels, valid_preds)
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_bce = np.mean(valid_bce_losses)
        avg_valid_reg = np.mean(valid_reg_losses)

        # ========== History ==========
        history['train_loss'].append(np.mean(train_losses))
        history['train_metrics'].append(train_metrics)
        history['valid_loss'].append(np.mean(valid_losses))
        history['valid_metrics'].append(valid_metrics)
        history['reg_loss'].append(np.mean(train_reg_losses))
        feature_importance = model.get_feature_importance()
        history['feature_importance'].append(feature_importance)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        # ========== Progress prints ==========
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs} (LR: {current_lr:.2e})")
            print("-"*60)
            print(f"Loss - Train: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Reg: {avg_train_reg:.4f})")
            print(f"       Valid: {avg_valid_loss:.4f} (BCE: {avg_valid_bce:.4f}, Reg: {avg_valid_reg:.4f})")
            print(f"\nTrain metrics:")
            print(f"  F1: {train_metrics['f1']:.4f}, Accuracy: {train_metrics['accuracy']:.4f},Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}")
            print(f"\nValid metrics:")
            print(f"  F1: {valid_metrics['f1']:.4f}, Accuracy: {valid_metrics['accuracy']:.4f},Precision: {valid_metrics['precision']:.4f}, "
                  f"Recall: {valid_metrics['recall']:.4f}")
            
            # # Active group stats
            # important_groups = sum(1 for score in feature_importance.values() if score > 0.1)
            # print(f"\nActive feature groups: {important_groups}/{len(feature_groups)}")

            # # Active group stats
            # important_groups = sum(1 for score in feature_importance.values() if score > 0.1)

        # Early stopping
        if valid_metrics['f1'] > best_valid_f1:
            best_valid_f1 = valid_metrics['f1']
            best_model_state = model.state_dict().copy()
            best_metrics = valid_metrics.copy()  
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"New best valid F1: {best_valid_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: no improvement in {patience} epochs")
                print(f"Best model from epoch {best_epoch}")
                break

        # Severe overfitting check
        if epoch > 10:
            overfitting_gap = train_metrics['f1'] - valid_metrics['f1']
            if overfitting_gap > 0.3:
                print(f"Warning: severe overfitting detected (gap: {overfitting_gap:.4f})")

    # ========== Restore best ==========
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (Epoch {best_epoch})")

    # Fallback for best_metrics
    if best_metrics is None:
        best_metrics = valid_metrics

    # 简洁 summary
    print("\n" + "="*60)
    print(f"Training complete (Best epoch {best_epoch})")
    print_detailed_metrics(best_metrics)

    return model, best_valid_f1, best_metrics, history, best_epoch

# COMMAND ----------

def train_model_with_group_lasso(train_df, valid_df, feature_cols, 
                                 feature_groups=None,
                                 n_epochs=30, batch_size=32, 
                                 lr=1e-3, 
                                 lambda_l2=0.01,
                                 lambda_l1=0.001,
                                 use_class_weights=True,
                                 patience=10,
                                 retrain_with_selected=True,
                                 use_focal_loss=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clean feature columns
    feature_cols_clean = [col for col in feature_cols 
                          if col not in ['sample_id', '_seq_position', 'Tc_target', 
                                       'has_last_fail', 'days_since_last_fail', 'Wagon_ICWVehicle',
                                       'has_data','doy_cos','doy_sin','time_position','days_to_target']]
    
    print(f"Num features after cleaning: {len(feature_cols_clean)}")

    # Auto-create feature groups (if not provided)
    if feature_groups is None:
        # Group by name prefix
        feature_groups = create_feature_groups_by_prefix(feature_cols_clean, prefix_dict=prefix_dict_with_group)
        # feature_groups = create_feature_groups_by_prefix(feature_cols_clean, prefix_dict=prefix_dict_no_group)
        print(f"Auto-created {len(feature_groups)} feature groups")
    
    # Create datasets
    train_dataset = FastPandasDataset(train_df, feature_cols_clean,label_col='Tc_target',
                                  sample_col='sample_id', seq_col='_seq_position',clip_mode='percentile',
                                  clip_percentile=(0.5, 99.5),clip_fixed=(-10.0, 10.0))

    valid_dataset = FastPandasDataset(valid_df, feature_cols_clean,label_col='Tc_target',
                                  sample_col='sample_id', seq_col='_seq_position',clip_mode='percentile',
                                  clip_percentile=(0.5, 99.5),clip_fixed=(-10.0, 10.0))


    # Compute class weights
    if use_class_weights:
        pos_count = (train_dataset.y == 1).sum()
        neg_count = (train_dataset.y == 0).sum()
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"Positive class weight: {pos_weight:.2f}")
        print(f"Train set - Positives: {pos_count}, Negatives: {neg_count}")
    else:
        pos_weight = 1.0
    
    # Create dataloaders (no oversampling)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    input_dim = len(feature_cols_clean)

    model = GroupLassoTransformerModel(
        input_dim=input_dim,
        d_model=96,
        nhead=4,
        num_layers=1,
        dim_feedforward=192,
        dropout=0.1,  
        feature_groups=feature_groups
    )
    model = model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Loss & optimizer
    if use_focal_loss:
        # Use Focal Loss; pos_weight not needed
        criterion = FocalLoss(alpha=0.75, gamma=1.0)
        print("Using Focal Loss (alpha=0.75,, gamma=1.0)")
    else:
        # Weighted BCE
        if use_class_weights:
            pos_weight_value = max(1.0, neg_count / max(1, pos_count))  # ~3.1
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
            )
            print("Using use_class_weights with BCEWithLogitsLoss")
        else:
            pos_weight = 1.0
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))
            print("Not Using use_class_weights with BCEWithLogitsLoss")


    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if (p.ndim == 1) or ("bias" in n) or ("LayerNorm" in n) or ("layernorm" in n):
            no_decay.append(p)   # LN weights & biases -> no weight decay
        else:
            no_decay_flag = False
            no_decay_flag = no_decay_flag  # placeholder for extra exclusions
            decay.append(p)

    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-5},{"params": no_decay, "weight_decay": 0.0}],lr=2e-4, betas=(0.9, 0.99))   

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, n_epochs)
    lr_warmup_steps = min(500, max(1, total_steps // 10))  # first 10% (cap at 500) warmup

    # LR scheduler
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < lr_warmup_steps:
            return float(current_step) / float(max(1, lr_warmup_steps))
        # then linearly decay to 0
        progress = (current_step - lr_warmup_steps) / float(max(1, total_steps - lr_warmup_steps))
        return max(0.0, 1.0 - progress)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # First training
    model, best_valid_f1, best_metrics, history, best_epoch = train_loop(
        model, train_loader, valid_loader, optimizer, scheduler, criterion,
        n_epochs=n_epochs, patience=patience,
        lambda_l2=lambda_l2, lambda_l1=lambda_l1, device=device
    )
        
    # ========== Feature importance ==========
    final_importance = model.get_feature_importance()
    
    print("\n" + "="*60)
    print("Feature selection results")
    print("="*60)
    
    # Map importance to names
    feature_importance_with_names = []
    for i, col in enumerate(feature_cols_clean):
        if i in final_importance:
            feature_importance_with_names.append((col, final_importance[i]))
    
    # Sort & print
    feature_importance_with_names.sort(key=lambda x: x[1], reverse=True)
    
    # Print top features
    print("\nTop important features:")
    for i, (name, score) in enumerate(feature_importance_with_names[::]):
        print(f"{i+1:3d}. {name:40s} importance: {score:.4f}")
    
    # Selection stats
    # threshold = 0.65
    # selected_features = [name for name, score in feature_importance_with_names if score > threshold]
    # print(f"  Selection threshold: {threshold}")

    top_K = 8  # you can change K here
    selected_features = [name for name, score in feature_importance_with_names[:top_K]]
    print(f"  Selection top_K: {top_K}")

    print(f"\nFeature selection stats:")
    print(f"  Selected features: {len(selected_features)}/{len(feature_cols_clean)}")
    print(f"  Selection ratio: {len(selected_features)/len(feature_cols_clean):.1%}")
    
    # Retrain with selected features
    if retrain_with_selected and len(selected_features) > 0:
        print("\nRetraining with selected features only...")
        model, best_valid_f1, best_metrics, history, selected_features = train_model_with_group_lasso(train_df, 
                                     valid_df, 
                                     selected_features,
                                     feature_groups=None,
                                     n_epochs=n_epochs, 
                                     batch_size=batch_size,
                                     lr=lr, 
                                     patience=patience,
                                     lambda_l2=lambda_l2, 
                                     lambda_l1=lambda_l1,
                                     use_focal_loss=True,
                                     retrain_with_selected=False)

    # ========= Final Summary =========
    print("\n" + "="*60)
    print("FINAL TRAINING SUMMARY")
    print("="*60)
    print(f"Best epoch: {best_epoch}")
    print_detailed_metrics(best_metrics)

    if history['train_metrics']:
        final_train_f1 = history['train_metrics'][-1]['f1']
        final_valid_f1 = history['valid_metrics'][-1]['f1']
        gap = final_train_f1 - final_valid_f1
        print(f"\nOverfitting analysis:")
        print(f"  Final Train F1: {final_train_f1:.4f}")
        print(f"  Final Valid F1: {final_valid_f1:.4f}")
        print(f"  Gap: {gap:.4f}")
        if gap < 0.1:
            print("  Status: !!!Good")
        elif gap < 0.2:
            print("  Status: !!!Mild overfitting")
        else:
            print("  Status: !!!Severe overfitting")

    return model, best_valid_f1, best_metrics, history, selected_features

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd

class TorchModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model, feature_names, threshold=0.5, 
                 device="cpu", strict=True,sample_col="sample_id", 
                 seq_col="_seq_position", pkey_col="Tc_p_key",
                 clip_mode="percentile", # "percentile" / "fixed" / "none"
                 clip_percentile=(0.5, 99.5),
                 clip_fixed=(-10.0, 10.0)):

        self.model = model.to(device).eval()
        self.threshold = threshold
        self.device = device
        self.feature_names_in_ = feature_names   # keep training feature column names

        self.sample_col = sample_col
        self.seq_col = seq_col
        self.pkey_col = pkey_col 

        self.feature_cols = []
        self.clip_mode = clip_mode
        self.clip_percentile = clip_percentile
        self.clip_fixed = clip_fixed
        self.strict = strict

    def _build_3d_from_df(self, model_input_df: pd.DataFrame):

        meta_cols = {self.sample_col, self.seq_col}
        self.feature_cols = [c for c in self.feature_names_in_ if c not in meta_cols]
        leaked = [c for c in feature_cols if c in meta_cols]
        if leaked:
            print(f"[Warn] Removed metadata columns from feature_cols: {leaked}")

        need = [self.sample_col] + self.feature_cols
        df_cols = model_input_df.columns
        missing = [c for c in need if c not in df_cols]
        if missing and self.strict:
            raise KeyError(f"Missing columns: {missing}")
        elif missing:
            print(f"[Warn] Missing columns will be ignored: {missing}")
            need = [c for c in need if c in df_cols]

        # (2) Sorting (match the original: only by sample_id, preserve within-group row order)
        extra = [self.pkey_col] if self.pkey_col in df_cols else []
        pdf = model_input_df[need + extra].copy()
        # Stable sort by sample_id only (consistent with the current Dataset)
        pdf.sort_values([self.sample_col], inplace=True, kind="mergesort")

        # (3) Group -> (T, F)
        X_list, keys = [], []
        for sid, g in pdf.groupby(self.sample_col, sort=False):
            # Match the original: select only feature columns -> numpy(float32) -> replace nan/inf with 0
            x_np = g[self.feature_names_in_].to_numpy(dtype=np.float32)
            x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
            X_list.append(torch.from_numpy(x_np))
            # Return key (prefer p_key, otherwise sample_id)
            if self.pkey_col in g.columns:
                keys.append(g[self.pkey_col].iloc[0])
            else:
                keys.append(sid)

        # (N, T, F)
        all_samples = np.stack([x.numpy() for x in X_list]) if len(X_list) > 0 else \
                      np.zeros((0, 0, len(self.feature_names_in_)), dtype=np.float32)

        # (4) Align with FastPandasDataset "conditional standardization"
        if all_samples.size > 0:
            N, T, F = all_samples.shape
            for i in range(F):
                feature_data = all_samples[:, :, i]
                mean = np.mean(feature_data)
                std = np.std(feature_data) + 1e-6
                # Threshold can be tuned; same as training side
                if std > 10 or abs(mean) > 10:
                    feature_data = (feature_data - mean) / std
                    all_samples[:, :, i] = feature_data

            # (5) Clipping (consistent with training)
            if self.clip_mode == "percentile":
                p_low, p_high = self.clip_percentile
                for i in range(F):
                    feat = all_samples[:, :, i]
                    lo, hi = np.percentile(feat, [p_low, p_high])
                    if lo < hi:  # avoid lo==hi causing zeroing out
                        all_samples[:, :, i] = np.clip(feat, lo, hi)
            elif self.clip_mode == "fixed":
                lo, hi = self.clip_fixed
                all_samples = np.clip(all_samples, lo, hi)
            elif self.clip_mode == "none":
                pass
            else:
                raise ValueError(f"Unknown clip_mode: {self.clip_mode}")

        return all_samples.astype(np.float32), keys
    
    def predict(self, context, model_input_df: pd.DataFrame):
        _ = context  # MLflow requires this argument; unused

        # Build (N, T, F)
        X_np, keys = self._build_3d_from_df(model_input_df)

        if X_np.size == 0:
            return pd.DataFrame(columns=["p_key", "target", "probability"])
        
        # Run model
        with torch.no_grad():
            x_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            logits = self.model(x_t)            # expected shape: (N,)
            if not torch.isfinite(logits).all():
                print("!!!!Wrapper detected non-finite outputs; returning zeros")
                logits = torch.zeros_like(logits)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        labels = (probs >= self.threshold).astype(int)

        # One row per sample
        out = pd.DataFrame({
            "p_key": keys,
            "target": labels,
            "probability": probs
        })

        return out

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def _collect_probs_labels(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, m = batch
                x, y, m = x.to(device), y.squeeze().to(device), m.to(device)
                logits = model(x, mask=m, training=False)
            else:
                x, y = batch
                x, y = x.to(device), y.squeeze().to(device)
                logits = model(x, training=False)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())
    return np.asarray(labels).astype(int), np.asarray(probs).astype(float)


def pick_threshold(
    y_true, y_prob,
    policy="f1",                    # 'f1' | 'youden' | 'quantile' | 'precision_at' | 'recall_at' | 'cost'
    target_pos_rate=None,           # for f1 alignment / quantile: target positive rate
    target_value=None,              # used by precision_at / recall_at / cost
    fp_cost=None, fn_cost=None,     # used by cost: unit cost for FP/FN
    grid_size=401,                  # number of thresholds to scan
    min_pos_rate=None, 
    max_pos_rate=None):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    lo, hi = float(y_prob.min()), float(y_prob.max())
    if lo == hi:
        # All probabilities are identical; cannot form a useful classifier
        t = hi
        # Unified return format: t, f1, p, r, lo, hi, steps, pos_rate
        pos_rate = 1.0  # predict all positive
        return t, 0.0, 0.0, 0.0, lo, hi, 1, pos_rate

    grid = np.linspace(lo, hi, grid_size)

    def stats_at(t):
        yp = (y_prob >= t).astype(int)
        tp = ((yp == 1) & (y_true == 1)).sum()
        fp = ((yp == 1) & (y_true == 0)).sum()
        tn = ((yp == 0) & (y_true == 0)).sum()
        fn = ((yp == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
        tpr  = rec
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        posr = yp.mean()
        return dict(t=t, f1=f1, prec=prec, rec=rec, tpr=tpr, fpr=fpr, pos_rate=posr,
                    tp=tp, fp=fp, tn=tn, fn=fn)

    stats_list = [stats_at(t) for t in grid]

    # Choose candidate threshold
    if policy == "f1":
        # Primary objective: maximize F1
        best = max(stats_list, key=lambda s: s["f1"])
        # If target_pos_rate is given, pick among the ">= 90% of peak F1" candidates
        # the one whose positive rate is closest to target_pos_rate
        if target_pos_rate is not None:
            peak = best["f1"]
            cands = [s for s in stats_list if s["f1"] >= 0.9 * peak]
            if cands:
                best = min(cands, key=lambda s: abs(s["pos_rate"] - target_pos_rate))
        cand = best

    elif policy == "youden":
        cand = max(stats_list, key=lambda s: (s["tpr"] - s["fpr"]))

    elif policy == "quantile":
        assert target_pos_rate is not None, "policy='quantile' requires target_pos_rate"
        t_q = float(np.quantile(y_prob, 1 - target_pos_rate))
        cand = stats_at(t_q)

    elif policy == "precision_at":
        assert target_value is not None, "policy='precision_at' requires target_value"
        feas = [s for s in stats_list if s["prec"] >= target_value]
        cand = (max(feas, key=lambda s: s["rec"]) if feas else max(stats_list, key=lambda s: s["prec"]))

    elif policy == "recall_at":
        assert target_value is not None, "policy='recall_at' requires target_value"
        feas = [s for s in stats_list if s["rec"] >= target_value]
        cand = (max(feas, key=lambda s: s["prec"]) if feas else max(stats_list, key=lambda s: s["rec"]))

    elif policy == "cost":
        assert fp_cost is not None and fn_cost is not None, "policy='cost' requires fp_cost and fn_cost"
        def expected_cost(s): return s["fp"] * fp_cost + s["fn"] * fn_cost
        cand = min(stats_list, key=expected_cost)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    if (min_pos_rate is not None) or (max_pos_rate is not None):
        s0 = cand
        violate = ((min_pos_rate is not None and s0["pos_rate"] < min_pos_rate) or
                   (max_pos_rate is not None and s0["pos_rate"] > max_pos_rate))
        if violate:
            # Find the point in the global grid closest to the band
            def distance_to_band(posr):
                d1 = 0.0 if (min_pos_rate is None or posr >= min_pos_rate) else (min_pos_rate - posr)
                d2 = 0.0 if (max_pos_rate is None or posr <= max_pos_rate) else (posr - max_pos_rate)
                return d1 + d2
            cand = min(stats_list, key=lambda s: distance_to_band(s["pos_rate"]))

    # Unified return format: t*, f1, p, r, lo, hi, steps, pos_rate
    return float(cand["t"]), float(cand["f1"]), float(cand["prec"]), float(cand["rec"]), lo, hi, grid_size, float(cand["pos_rate"])





# COMMAND ----------


def evaluate_and_register_model(model, test_loader, test_df, valid_loader, feature_cols, model_name, 
                                model_type='transformer', 
                                device='cuda',
                                ml_catalog='09ad024f-822f-48e4-9d9e-b5e03c1839a2', 
                                model_schema_name='feature_selection'):

    # ========== 1. Evaluation ==========
    print("\n" + "="*60)
    print("Final model evaluation")
    print("="*60)

    yv_true, yv_probs = _collect_probs_labels(model, valid_loader, device)
    valid_pos_rate = yv_true.mean()
    # Method 1: quantile alignment (stable, predicted positive rate ≈ actual positive rate on validation)
    t_quantile = float(np.quantile(yv_probs, 1 - valid_pos_rate))

    # Method 2: your adaptive function (primary objective = F1), using validation data
    t_adapt, f1_v, p_v, r_v, vlo, vhi, vsteps, vpos = pick_threshold(
        yv_true, yv_probs,
        policy="f1",                 #"quantile"/"youden"/...
        target_pos_rate=valid_pos_rate,
        min_pos_rate=valid_pos_rate*0.6,
        max_pos_rate=valid_pos_rate*1.4
    )

    # Choose a preferred strategy (recommend adaptive + alignment)
    best_threshold = t_adapt
    print(f"[VALID] actual_pos_rate={valid_pos_rate:.3f} | t_quantile={t_quantile:.4f} | t_adapt={t_adapt:.4f} (F1={f1_v:.4f}, P={p_v:.4f}, R={r_v:.4f}, pred_pos_rate≈{vpos:.3f})")

    # ========= B. Final evaluation on test set with the fixed threshold =========
    y_true, y_probs = _collect_probs_labels(model, test_loader, device)
    y_pred = (y_probs >= best_threshold).astype(int)

    print(f"\nTest sample stats: total={len(y_true)}, positives={np.sum(y_true)}, negatives={len(y_true)-np.sum(y_true)}")
    print(f"Prediction probability distribution: mean={y_probs.mean():.4f}, std={y_probs.std():.4f}")
    print(f"Prediction probability range: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
    print(f"best_threshold: {best_threshold}")
    q = np.quantile(y_probs, [0.0,0.01,0.05,0.10,0.25,0.5,0.75,0.90,0.95,0.99,1.0])
    print("Probability quantiles:", dict(zip(["min","p1","p5","p10","p25","p50","p75","p90","p95","p99","max"], q)))

    # calculate_metrics expects probabilities + a fixed threshold
    metrics = calculate_metrics(y_true, y_probs, threshold=best_threshold)
    print("\nFinal test performance (using validation-set threshold):")
    print_detailed_metrics(metrics)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    pred_pos_rate = y_pred.mean()
    actual_pos_rate = y_true.mean()
    print("\nConfusion matrix [ [TN FP], [FN TP] ] :")
    print(cm)
    print(f"Actual positive rate: {actual_pos_rate:.3f}")
    print(f"Predicted positive rate: {pred_pos_rate:.3f}")
    print(f"Prediction bias: {(pred_pos_rate - actual_pos_rate)/max(1e-8, actual_pos_rate)*100:+.1f}%")
    print(f"\nOverall accuracy (Accuracy): {metrics['accuracy']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=['No Failure','Failure'], digits=4, zero_division=0))

    print("\nDiagnostics:")
    print(f"- Fraction with prob > 0.9: {(y_probs > 0.9).mean():.3f}")
    print(f"- Fraction with prob < 0.1: {(y_probs < 0.1).mean():.3f}")
    print(f"- TPR among high-confidence positives: {y_true[y_probs > 0.8].mean():.3f}" if (y_probs > 0.8).any() else "- No high-confidence positives")
    print(f"- TNR among high-confidence negatives: {1-y_true[y_probs < 0.2].mean():.3f}" if (y_probs < 0.2).any() else "-- No high-confidence negatives")

    registered_model_name=f"{ml_catalog}.{model_schema_name}.{model_name}"

    ####################################
    # Evaluate the model and register to MLflow
    ####################################
    with mlflow.start_run() as run:
        mlflow.log_param("best_threshold", float(best_threshold))
        # Take a small sample as example input
        sample_input = test_df.select("Tc_BaseCode","Tc_BaseCode_Mapped","Tc_SectionBreakStartKM","Tc_date","sample_id","Tc_p_key","Wagon_Twist14m","Wagon_BounceFrt","Wagon_BounceRr","Wagon_BodyRockFrt","Wagon_BodyRockRr","Wagon_LP1","Wagon_LP2","Wagon_LP3","Wagon_LP4","Wagon_Speed","Wagon_BrakeCylinder","Wagon_IntrainForce","Wagon_Acc1","Wagon_Acc2","Wagon_Acc3","Wagon_Acc4","Wagon_Twist2m","Wagon_Acc1_RMS","Wagon_Acc2_RMS","Wagon_Acc3_RMS","Wagon_Acc4_RMS","Wagon_Rail_Pro_L","Wagon_Rail_Pro_R","Wagon_SND","Wagon_VACC","Wagon_VACC_L","Wagon_VACC_R","Wagon_Curvature","Wagon_Track_Offset","Wagon_ICWVehicle","Wagon_SND_L","Wagon_SND_R").toPandas().fillna(0).head(5)


        sample_output = TorchModelWrapper(model, feature_cols, best_threshold,device="cpu",sample_col="sample_id",pkey_col="Tc_p_key",clip_mode="percentile",clip_percentile=(0.5, 99.5),clip_fixed=(-10.0, 10.0)).predict(
            None, sample_input
        )
        sign = infer_signature(sample_input, sample_output)
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=TorchModelWrapper(
                model=model,
                feature_names=feature_cols,
                threshold=best_threshold
            ),
            registered_model_name=registered_model_name,
            signature=sign
        )

        run_id = run.info.run_id
        print(f"\n !!!!Model registered to MLflow")
        print(f"   Run ID: {run_id}")
        print(f"   Model Name: {ml_catalog}.{model_schema_name}.{model_name}")

    # # ========== 4. Create results DataFrame ==========
    # Prepare result metrics
    model_metrics = {
        "accuracy": float(metrics['accuracy']),
        "f1_score": float(metrics['f1']),
        "precision": float(metrics['precision']),
        "recall": float(metrics['recall']),
        "roc_auc": float(metrics['roc_auc']) if not np.isnan(metrics['roc_auc']) else None,
        "pr_auc": float(metrics['pr_auc']) if not np.isnan(metrics['pr_auc']) else None,
        "threshold": float(best_threshold),
        "tp": int(metrics['tp']),
        "fp": int(metrics['fp']),
        "tn": int(metrics['tn']),
        "fn": int(metrics['fn'])
    }
    
    # Create Pandas DataFrame
    df_result_pd = pd.DataFrame(
        data=[[registered_model_name,1,str({"accuracy":metrics['accuracy']}),0]],
        columns=['ModelName', 'ModelVersion', 'ModelMetrics', 'PipelineVersion']
    )

    # Convert to Spark DataFrame
    df_result = spark.createDataFrame(df_result_pd)
    
    print("\n  Result DataFrame:")
    df_result.show(truncate=False)

    return df_result, model_metrics, run_id

# COMMAND ----------

#========== 5. Integrate into the main training pipeline ==========
def train_and_register_best_model(train_df, valid_df, test_df, 
                                  feature_cols,n_epochs=10, batch_size=32,
                                  model_name="transformer_group_lasso_test_v2"):
    """
    Train the model and register the best version
    """
    # Train the model (using your previous training function)
    print("Training model...")
    print("Cleaning and validating feature columns...")
    # Clean feature columns
    feature_cols_clean = [col for col in feature_cols 
                          if col not in ['sample_id', '_seq_position', 'Tc_target', 
                                       'has_last_fail', 'days_since_last_fail', 'Wagon_ICWVehicle',
                                       'has_data','doy_cos','doy_sin','time_position','days_to_target']]
    
    print(f"Number of features after cleaning: {len(feature_cols_clean)}")
    print(f"Feature sample: {feature_cols_clean[::]}")

    # Optionally use the Group Lasso version
    model, best_f1, best_metrics, history, selected_features = train_model_with_group_lasso(
        train_df, valid_df,
        feature_cols,
        feature_groups=None,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=1e-4,
        lambda_l2=0.0001,  # L2 regularization (tuned)
        lambda_l1=0,  # L1 regularization (tuned)
        patience=10,
        use_class_weights=True,  # Use class weights for balancing
        use_focal_loss=True,
        retrain_with_selected=True
    )

    # Create test data loader
    print("\nPreparing test data...")   

    final_feature_cols = [col for col in feature_cols 
                          if col not in ['sample_id', '_seq_position', 'Tc_target', 
                                       'has_last_fail', 'days_since_last_fail', 'Wagon_ICWVehicle',
                                       'has_data','doy_cos','doy_sin','time_position','days_to_target']]

    print(f"Final number of test features: {len(final_feature_cols)}")

    if len(final_feature_cols) == 0:
        print("!!!!Error: No valid test features")
        return None, None, None

    # 5. Create test/valid data loaders
    try:
        test_dataset = FastPandasDataset(test_df, final_feature_cols,label_col='Tc_target',
                                  sample_col='sample_id', seq_col='_seq_position',clip_mode='percentile',
                                  clip_percentile=(0.5, 99.5),clip_fixed=(-10.0, 10.0))
        test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=True)

        valid_dataset = FastPandasDataset(valid_df, feature_cols_clean,label_col='Tc_target',
                                  sample_col='sample_id', seq_col='_seq_position',clip_mode='percentile',
                                  clip_percentile=(0.5, 99.5),clip_fixed=(-10.0, 10.0))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    except Exception as e:
        print(f"Failed to create test data loader: {e}")
        print(f"Tried features: {final_feature_cols[::]}...")
        return None, None, None

    # Evaluate and register the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    df_result, final_metrics, run_id = evaluate_and_register_model(
        model=model,
        test_loader=test_loader,
        test_df=test_df,
        valid_loader=valid_loader,
        feature_cols=final_feature_cols,
        model_name=model_name,
        model_type='transformer_group_lasso_test_v2',
        device=device,
        ml_catalog='09ad024f-822f-48e4-9d9e-b5e03c1839a2',
        model_schema_name='feature_selection'
    )
    
    return model, df_result, final_metrics

# COMMAND ----------

# ========== 6. Run the end-to-end pipeline ==========
from pyspark.sql import functions as F

def split_data_three_way(expanded_df):
    """
    Split data into train, validation, and test sets
    """
    # Get unique dates and sort
    unique_dates = expanded_df.select("Tc_date").distinct().orderBy("Tc_date").collect()
    dates = [row['Tc_date'] for row in unique_dates]
    
    n_dates = len(dates)
    train_cutoff = dates[int(n_dates * 0.8)]
    valid_cutoff = dates[int(n_dates * 0.9)]
    
    train_df = expanded_df.filter(F.col("Tc_date") <= train_cutoff)
    valid_df = expanded_df.filter(
        (F.col("Tc_date") > train_cutoff) & 
        (F.col("Tc_date") <= valid_cutoff)
    )
    test_df = expanded_df.filter(F.col("Tc_date") > valid_cutoff)
    
    return train_df, valid_df, test_df

# Run the full pipeline
print("Starting the full model training and registration pipeline...")

# Split the data
train_df, valid_df, test_df = split_data_three_way(aggregated_df)

# Print dataset stats
for name, df in [("Train", train_df), ("Validation", valid_df), ("Test", test_df)]:
    samples = df.select("sample_id").distinct().count()
    pos_samples = df.select("sample_id", "Tc_target").distinct().filter(
        F.col("Tc_target") == 1
    ).count()
    print(f"{name}: {samples} samples (positives: {pos_samples}, {pos_samples/samples:.2%})")

# Train and register the model
model, df_result, final_metrics = train_and_register_best_model(
    train_df, valid_df, test_df,
    all_features,n_epochs=10, batch_size=128,
    model_name="transformer_group_lasso_test_v2"
)

print("\n------Done! The model has been trained, evaluated, and registered to MLflow")
print(f"Final accuracy: {final_metrics['accuracy']:.2%}")
print(f"Final F1-score: {final_metrics['f1_score']:.4f}")


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

# MAGIC %md ### Testing model
# MAGIC
# MAGIC you can test and create a subset of the training set for your testing.

# COMMAND ----------

# MAGIC %md ## Store model to model registry
# MAGIC Mlflow is the model registry that is used for storing and maintaing ML and AI models. We as insightfactory use it for storing and managing our models.
# MAGIC
# MAGIC This is just an example of how you can store model into mlflow. please find more docs about mlflow online [here](https://mlflow.org/docs/latest/introduction/index.html)

# COMMAND ----------

# import mlflow
# from mlflow.models.signature import infer_signature 

# with mlflow.start_run() as run:
#     ## create signature of the model input and output
#     sign=infer_signature(model_input=X_test.reset_index(drop=True),model_output=y_pred)

#     ## store the model using mlflow
#     mlflow.sklearn.log_model(rf_classifier,model_name
#                              ,registered_model_name=f'{ml_catalog}.{model_schema_name}.{model_name}',
#                              signature=sign)

# COMMAND ----------

################# Update your output data for the model configuration here #################
# import pandas as pd
# df_result=spark.createDataFrame(pd.DataFrame(
#     data=[[model_name,1,str({"accuracy":accuracy}),0]],
#     columns=['ModelName','ModelVersion','ModelMetrics','PipelineVersion']
# ))

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

# MAGIC %sql
# MAGIC
# MAGIC SELECT count(*)  FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.feature_selection.total_training_table
