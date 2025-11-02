# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Feature engineering dataset
# MAGIC
# MAGIC **Business Rules:** <br/>
# MAGIC \<Describe the Business Rules that are encapsulated in this Enrichment\>
# MAGIC
# MAGIC **Dependencies:**<br/>
# MAGIC \<List the dependencies that need to be satisfied before this Enrichment can execute\>
# MAGIC
# MAGIC **Ownership:**<br/>
# MAGIC \<Indicate who owns this Enrichment ruleset\>

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC #### Modification Schedule
# MAGIC
# MAGIC | Date | Who | Description |
# MAGIC | ---: | :--- | :--- |
# MAGIC | 2025-09-01 | Zi Lun Ma | Initial version. |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Insight Factory Notebook Preparation
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**

# COMMAND ----------

# MAGIC %run "/InsightFactory/Helpers/Update Delta Lake Preparation"

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
# MAGIC - If your cells below are Python, ensure that the result is stored in a Spark dataframe called 'df_result' e.g. df_result = ...
# MAGIC - If your cells below are SQL, the result of the SQL will be automatically processed (i.e. there is no need for you to do a formal assignment).
# MAGIC <br/><br/>
# MAGIC
# MAGIC ### Running this Notebook directly in Databricks
# MAGIC
# MAGIC This Notebook can be run directly from your Databricks Workspace.  If the Notebook relies on Notebook Parameters, please read the following instructions:
# MAGIC 1) Add this line of code to a cell at the top of your Notebook and run that cell.<br/>
# MAGIC    ```dbutils.widgets.text('ParametersJSON', '{ "NotebookParameters": { "param1": "value1", "param2": "value2" } }')```
# MAGIC 2) This will add a Parameter to the Notebook.  Simply replace (or remove) the pre-canned parameters, 'param1', 'param2' and their values with your own.
# MAGIC 3) When you have finished running this Notebook directly in Databricks, comment out the line of code you added or delete the cell entirely.    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Training dataset 
# MAGIC This space is for you to create features that you will use in your model. Feature engineering and selection can be time consuming and could be something you might look to spend more time on. This sets the base for your models. You can create models like Time Series, Moving Averages, Fourier transforms, Complex Transforms, Statistical measures and various other features. 
# MAGIC
# MAGIC
# MAGIC Have a reasearch and Happy Coding them.
# MAGIC
# MAGIC **Notes: you can install factory_ml as package which contains alot of feature engineering methods that you can utilize. Just install by `pip install factory_ml` and run `dbutils.library.restartPython()` and start using with dir(factory_ml) and help(factory_ml).**

# COMMAND ----------

context = params.context
if context == "testing":
    base_table = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.preprocess_testing_table"
    threshold_table = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.feature_engineering.fe_thresholds_test"
else:
    # default to training
    base_table = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.preprocess_training_table"
    threshold_table = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.feature_engineering.fe_thresholds_train"

df_result = spark.sql(
f"""
WITH base AS (
  SELECT *, 
  YEAR(Tc_r_date) AS Year_Partition, 
  MONTH(Tc_r_date) AS Month_Partition
  FROM {base_table}
),
risk_spikes AS (
  SELECT
    Tc_BaseCode, 
    Tc_SectionBreakStartKM,
    Year_Partition,
    Month_Partition,
    SUM(CASE WHEN Wagon_Twist14m > Threshold_Twist14m THEN 1 ELSE 0 END) AS fe_Wagon_Twist14m_Spike,
    SUM(CASE WHEN Wagon_Twist2m > Threshold_Twist2m THEN 1 ELSE 0 END) AS fe_Wagon_Twist2m_Spike,
    SUM(CASE WHEN Wagon_BrakeCylinder > Threshold_BrakeCylinder THEN 1 ELSE 0 END) AS fe_Wagon_BrakeCylinder_Spike,
    SUM(CASE WHEN Wagon_IntrainForce > Threshold_IntrainForce THEN 1 ELSE 0 END) AS fe_Wagon_IntrainForce_Spike,
    SUM(CASE WHEN Wagon_SND > Threshold_SND THEN 1 ELSE 0 END) AS fe_Wagon_SND_Spike,
    SUM(CASE WHEN Wagon_SND_L > Threshold_SND_L THEN 1 ELSE 0 END) AS fe_Wagon_SND_L_Spike,
    SUM(CASE WHEN Wagon_SND_R > Threshold_SND_R THEN 1 ELSE 0 END) AS fe_Wagon_SND_R_Spike,
    SUM(CASE WHEN Wagon_LP1 > Threshold_LP1 THEN 1 ELSE 0 END) AS fe_Wagon_LP1_Spike,
    SUM(CASE WHEN Wagon_LP2 > Threshold_LP2 THEN 1 ELSE 0 END) AS fe_Wagon_LP2_Spike,
    SUM(CASE WHEN Wagon_LP3 > Threshold_LP3 THEN 1 ELSE 0 END) AS fe_Wagon_LP3_Spike,
    SUM(CASE WHEN Wagon_LP4 > Threshold_LP4 THEN 1 ELSE 0 END) AS fe_Wagon_LP4_Spike,
    SUM(CASE WHEN Wagon_Track_Offset > Threshold_Track_Offset THEN 1 ELSE 0 END) AS fe_Wagon_Track_Offset_Spike
  FROM base
  LEFT JOIN {threshold_table} USING (Tc_BaseCode, Tc_SectionBreakStartKM, Year_Partition, Month_Partition)
  GROUP BY Tc_BaseCode, Tc_SectionBreakStartKM, Year_Partition, Month_Partition
)

SELECT
  concat(Tc_BaseCode, '_', Tc_SectionBreakStartKM, '_20m_', Tc_r_date) AS p_key,
  {'Tc_target,' if context == 'training' else ''}
  Tc_BaseCode,
  Tc_SectionBreakStartKM,
  Tc_r_date,
  Wagon_RecordingDate,
  -- Bounce: Bounce measurement at the front/rear.
  Wagon_BounceFrt,
  COALESCE(ABS(Wagon_BounceFrt) / NULLIF(Threshold_BounceFrt, 0), 0) AS fe_Wagon_BounceFrt_Risk,
  Wagon_BounceRr,
  COALESCE(ABS(Wagon_BounceRr) / NULLIF(Threshold_BounceRr, 0), 0) AS fe_Wagon_BounceRr_Risk,
  -- Body rock: Body rock measurement at the front/rear.
  Wagon_BodyRockFrt,
  COALESCE(ABS(Wagon_BodyRockFrt) / NULLIF(Threshold_BodyRockFrt, 0), 0) AS fe_Wagon_BodyRockFrt_Risk,
  Wagon_BodyRockRr,
  COALESCE(ABS(Wagon_BodyRockRr) / NULLIF(Threshold_BodyRockRr, 0), 0) AS fe_Wagon_BodyRockRr_Risk,
  -- Load points: Load point measurement
  Wagon_LP1,
  Wagon_LP2,
  Wagon_LP3,
  Wagon_LP4,
  fe_Wagon_LP1_Spike,
  fe_Wagon_LP2_Spike,
  fe_Wagon_LP3_Spike,
  fe_Wagon_LP4_Spike,
  -- Speed
  Wagon_Speed,
  COALESCE(Wagon_Speed / NULLIF(Threshold_Speed, 0), 0) AS fe_Wagon_Speed_Risk,
  -- Twist: Average twist force applied to the cart over the last 14 or 2m
  Wagon_Twist2m,
  Wagon_Twist14m,
  fe_Wagon_Twist14m_Spike,
  fe_Wagon_Twist2m_Spike,
  -- Brake: Brake cylinder pressure
  Wagon_BrakeCylinder,
  fe_Wagon_BrakeCylinder_Spike,
  -- IntrainForce: Intrain force measurement
  Wagon_IntrainForce,
  fe_Wagon_IntrainForce_Spike,
  -- Acc: Acceleration measurements
  Wagon_Acc1,
  Wagon_Acc2,
  Wagon_Acc3,
  Wagon_Acc4,
  COALESCE(Wagon_Acc1 / NULLIF(Threshold_Acc1, 0), 0) AS fe_Wagon_Acc1_Risk,
  COALESCE(Wagon_Acc2 / NULLIF(Threshold_Acc2, 0), 0) AS fe_Wagon_Acc2_Risk,
  COALESCE(Wagon_Acc3 / NULLIF(Threshold_Acc3, 0), 0) AS fe_Wagon_Acc3_Risk,
  COALESCE(Wagon_Acc4 / NULLIF(Threshold_Acc4, 0), 0) AS fe_Wagon_Acc4_Risk,
  -- Acc_RMS: Root Mean Square of acceleration measurements
  Wagon_Acc1_RMS,
  Wagon_Acc2_RMS,
  Wagon_Acc3_RMS,
  Wagon_Acc4_RMS,
  COALESCE(ABS(Wagon_Acc1_RMS) / NULLIF(Threshold_Acc1_RMS, 0), 0) AS fe_Wagon_Acc1_RMS_Risk,
  COALESCE(ABS(Wagon_Acc2_RMS) / NULLIF(Threshold_Acc2_RMS, 0), 0) AS fe_Wagon_Acc2_RMS_Risk,
  COALESCE(ABS(Wagon_Acc3_RMS) / NULLIF(Threshold_Acc3_RMS, 0), 0) AS fe_Wagon_Acc3_RMS_Risk,
  COALESCE(ABS(Wagon_Acc4_RMS) / NULLIF(Threshold_Acc4_RMS, 0), 0) AS fe_Wagon_Acc4_RMS_Risk,
  -- Rail Pro: Rail profile measurements
  Wagon_Rail_Pro_L,
  Wagon_Rail_Pro_R,
  (Wagon_Rail_Pro_L - Wagon_Rail_Pro_R) AS fe_Wagon_Rail_Pro_LR_Diff,
  COALESCE(Wagon_Rail_Pro_L / NULLIF(Threshold_Rail_Pro, 0), 0) AS fe_Wagon_Rail_Pro_L_Risk,
  COALESCE(Wagon_Rail_Pro_R / NULLIF(Threshold_Rail_Pro, 0), 0) AS fe_Wagon_Rail_Pro_R_Risk,
  -- Sound
  Wagon_SND,
  Wagon_SND_L,
  Wagon_SND_R,
  (Wagon_SND_L - Wagon_SND_R) AS fe_Wagon_SND_LR_Diff,
  fe_Wagon_SND_Spike,
  fe_Wagon_SND_L_Spike,
  fe_Wagon_SND_R_Spike,
  -- VACC: Vertical acceleration measurement
  Wagon_VACC,
  Wagon_VACC_L,
  Wagon_VACC_R,
  COALESCE(Wagon_VACC / NULLIF(Threshold_VACC, 0), 0) AS fe_Wagon_VACC_Risk,
  COALESCE(Wagon_VACC_L / NULLIF(Threshold_VACC_L, 0), 0) AS fe_Wagon_VACC_L_Risk,
  COALESCE(Wagon_VACC_R / NULLIF(Threshold_VACC_R, 0), 0) AS fe_Wagon_VACC_R_Risk,
  (Wagon_VACC_L - Wagon_VACC_R) AS fe_VACC_LR_Diff,
  -- Curvature: Curvature of the track
  Wagon_Curvature,
  -- Offset: Offset of the track
  Wagon_Track_Offset,
  fe_Wagon_Track_Offset_Spike
  Wagon_ICWVehicle,
  --w_row_count,
  Tng_Tonnage
FROM base
LEFT JOIN {threshold_table} USING (Tc_BaseCode, Tc_SectionBreakStartKM, Year_Partition, Month_Partition)
LEFT JOIN risk_spikes USING (Tc_BaseCode, Tc_SectionBreakStartKM, Year_Partition, Month_Partition)
ORDER BY Tc_BaseCode, Tc_SectionBreakStartKM, Tc_r_date, Wagon_RecordingDate
""")

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

# Baseline features (Rolling averages)
# window for lag & rolling features
window_spec = Window.partitionBy("Tc_BaseCode", "Tc_SectionBreakStartKM").orderBy("Tc_r_date", "Wagon_RecordingDate")
df_result = df_result.withColumn("fe_Wagon_Speed_avg7", F.avg("Wagon_Speed").over(window_spec.rowsBetween(-6, 0)))
df_result = df_result.withColumn("fe_Wagon_Speed_avg30", F.avg("Wagon_Speed").over(window_spec.rowsBetween(-29, 0)))

# Interaction features
df_result = df_result.withColumn("fe_Curvature_x_Speed", F.col("Wagon_Curvature") * F.col("Wagon_Speed"))
df_result = df_result.withColumn("fe_Curvature_x_BrakeCylinder", F.col("Wagon_Curvature") * F.col("Wagon_BrakeCylinder"))
df_result = df_result.withColumn("fe_Speed_x_BrakeCylinder", F.col("Wagon_Speed") * F.col("Wagon_BrakeCylinder"))
df_result = df_result.withColumn("fe_Speed_x_Twist14m", F.col("Wagon_Speed") * F.col("Wagon_Twist14m"))
df_result = df_result.withColumn("fe_Acc1_x_Acc2", F.col("Wagon_Acc1") * F.col("Wagon_Acc2"))
df_result = df_result.withColumn("fe_Acc3_x_Acc4", F.col("Wagon_Acc3") * F.col("Wagon_Acc4"))
df_result = df_result.withColumn("fe_Twist2m_x_Twist14m", F.col("Wagon_Twist2m") * F.col("Wagon_Twist14m"))

# Lag features
lag_cols = [
    "Wagon_BounceFrt","Wagon_BounceRr","Wagon_BodyRockFrt","Wagon_BodyRockRr",
    "Wagon_LP1","Wagon_LP2","Wagon_LP3","Wagon_LP4",
    "Wagon_Speed", "Wagon_BrakeCylinder","Wagon_IntrainForce",
    "Wagon_Acc1","Wagon_Acc2","Wagon_Acc3","Wagon_Acc4",
    "Wagon_Acc1_RMS","Wagon_Acc2_RMS","Wagon_Acc3_RMS","Wagon_Acc4_RMS",
    "Wagon_Twist2m","Wagon_Twist14m",
    "Wagon_Rail_Pro_L","Wagon_Rail_Pro_R","Wagon_SND","Wagon_SND_L","Wagon_SND_R",
    "Wagon_VACC","Wagon_VACC_L","Wagon_VACC_R"
]

for col_name in lag_cols:
    for lag in [1,2,3]:
        df_result = df_result.withColumn(f"fe_{col_name}_lag{lag}", F.lag(col_name, lag).over(window_spec))
    # fill nulls with current value
    df_result = df_result.withColumn(f"fe_{col_name}_lag1", F.coalesce(F.col(f"fe_{col_name}_lag1"), F.col(col_name)))
    df_result = df_result.withColumn(f"fe_{col_name}_lag2", F.coalesce(F.col(f"fe_{col_name}_lag2"), F.col(f"fe_{col_name}_lag1")))
    df_result = df_result.withColumn(f"fe_{col_name}_lag3", F.coalesce(F.col(f"fe_{col_name}_lag3"), F.col(f"fe_{col_name}_lag2")))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook End
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**
# MAGIC
# MAGIC ####Important: 
# MAGIC Ensure that the result is stored in a Spark dataframe called 'df_result' e.g. df_result = ...

# COMMAND ----------

# MAGIC %run "/InsightFactory/Helpers/Update Delta Lake"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Testing/Debugging Zone
# MAGIC
# MAGIC All cells from here on are available to you to debug/test this notebook without impacting any task run.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- For testing purposes only

# COMMAND ----------

#df_result.select("*").limit(5).show()
