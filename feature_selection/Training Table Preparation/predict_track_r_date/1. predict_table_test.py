# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # training_table
# MAGIC
# MAGIC Creates your training table to be used for training the model
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
# MAGIC | 2025-08-30 | Sheng Wang| Initial version. |

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

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import types as T
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

# ========== 1) Read tables ==========
tc  = spark.table("dev_adlunise.predictive_maintenance_uofa_2025.testcontext")
w   = spark.table("dev_adlunise.predictive_maintenance_uofa_2025.wagondata")
bcm = spark.table("dev_adlunise.predictive_maintenance_uofa_2025.basecodemap")
tng = spark.table("`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.tonnagedata")

tc_df = (
  tc
  .filter(col("RecordingDate").isNotNull() & col("SectionBreakStartKM").isNotNull())
  .withColumn("Tc_r_date_minus_variable", F.date_sub(F.col("RecordingDate"), 30))
  .withColumn("km_range_start", F.col("SectionBreakStartKM").cast("double"))
  .withColumn("km_range_end",   (F.col("SectionBreakStartKM") + F.lit(0.02)).cast("double"))
  .join(bcm, on="BaseCode", how="left")
  .select(
      F.col("p_key").alias("Tc_p_key"),
      F.col("BaseCode").alias("Tc_BaseCode"),
      F.col("MappedBaseCode").alias("Tc_BaseCode_Mapped"),
      F.col("SectionBreakStartKM").alias("Tc_SectionBreakStartKM"),
      F.col("RecordingDate").alias("Tc_r_date"),
      "km_range_start", 
      "km_range_end",
      "Tc_r_date_minus_variable"
  )
  .withColumn(
      "Tc_window_end",F.date_add(F.to_date(F.col("Tc_r_date")), 0) 
  )
)


from pyspark.sql.functions import broadcast
tc_df = broadcast(tc_df)

# ========== 3) Wagon pre-aggregation: dimension reduction by “day × 0.02km × BaseCode(mapped/original)” ==========
w_df = (
  w.filter(col("RecordingDate").isNotNull() & col("SectionBreakStartKM").isNotNull())
   .join(bcm, on="BaseCode", how="left")
   .groupBy(
       F.col("BaseCode").alias("Wagon_BaseCode"),
       F.col("MappedBaseCode").alias("Wagon_BaseCode_Mapped"),
       F.col("RecordingDate").alias("Wagon_RecordingDate"),
       F.col("SectionBreakStartKM").alias("Wagon_SectionBreakStartKM"),
       F.col("SectionBreakFinishKM").alias("Wagon_SectionBreakFinishKM"),
   )
   .agg(
     F.avg("Twist14m").alias("Wagon_Twist14m"),
     F.avg("BounceFrt").alias("Wagon_BounceFrt"),
     F.avg("BounceRr").alias("Wagon_BounceRr"),
     F.avg("BodyRockFrt").alias("Wagon_BodyRockFrt"),
     F.avg("BodyRockRr").alias("Wagon_BodyRockRr"),
     F.avg("LP1").alias("Wagon_LP1"),
     F.avg("LP2").alias("Wagon_LP2"),
     F.avg("LP3").alias("Wagon_LP3"),
     F.avg("LP4").alias("Wagon_LP4"),
     F.avg("Speed").alias("Wagon_Speed"),
     F.avg("BrakeCylinder").alias("Wagon_BrakeCylinder"),
     F.avg("IntrainForce").alias("Wagon_IntrainForce"),
     F.avg("Acc1").alias("Wagon_Acc1"),
     F.avg("Acc2").alias("Wagon_Acc2"),
     F.avg("Acc3").alias("Wagon_Acc3"),
     F.avg("Acc4").alias("Wagon_Acc4"),
     F.avg("Twist2m").alias("Wagon_Twist2m"),
     F.avg("Acc1_RMS").alias("Wagon_Acc1_RMS"),
     F.avg("Acc2_RMS").alias("Wagon_Acc2_RMS"),
     F.avg("Acc3_RMS").alias("Wagon_Acc3_RMS"),
     F.avg("Acc4_RMS").alias("Wagon_Acc4_RMS"),
     F.avg("Rail_Pro_L").alias("Wagon_Rail_Pro_L"),
     F.avg("Rail_Pro_R").alias("Wagon_Rail_Pro_R"),
     F.avg("SND").alias("Wagon_SND"),
     F.avg("VACC").alias("Wagon_VACC"),
     F.avg("VACC_L").alias("Wagon_VACC_L"),
     F.avg("VACC_R").alias("Wagon_VACC_R"),
     F.avg("Curvature").alias("Wagon_Curvature"),
     F.avg("Track_Offset").alias("Wagon_Track_Offset"),
     F.avg("ICWVehicle").alias("Wagon_ICWVehicle"),
     F.avg("SND_L").alias("Wagon_SND_L"),
     F.avg("SND_R").alias("Wagon_SND_R"),
     F.count(F.lit(1)).alias("w_rows_in_bin")
   )
   .cache()
)


# ========== 4) Apply rough date-range filtering first to reduce w size ==========
# Use global min/max window bounds from tc to filter w (avoid full table scan)

tc_bounds = tc_df.select(
    F.min("Tc_r_date_minus_variable").alias("min_d"),
    F.max("Tc_window_end").alias("max_d")
).collect()[0]

w_df = w_df.filter((col("Wagon_RecordingDate") >= F.lit(tc_bounds["min_d"])) &
                   (col("Wagon_RecordingDate") <= F.lit(tc_bounds["max_d"]))).cache()

cond = (
    (F.coalesce(tc_df.Tc_BaseCode_Mapped, tc_df.Tc_BaseCode) ==
     F.coalesce(w_df.Wagon_BaseCode_Mapped, w_df.Wagon_BaseCode)) &
    (w_df.Wagon_RecordingDate >= tc_df.Tc_r_date_minus_variable) &
    (w_df.Wagon_RecordingDate <= tc_df.Tc_window_end) &
    (w_df.Wagon_SectionBreakStartKM <=  tc_df.km_range_end) &
    (w_df.Wagon_SectionBreakStartKM   >=  tc_df.km_range_start)
)

tc_aliased = tc_df.alias("tc")
w_aliased  = w_df.alias("w")

joined = (
    tc_aliased.join(w_aliased, cond, "inner")
    .select("tc.*", "w.*")
)

# display(joined)
# ========== 6) (Optional) Join Tonnage: interval & km overlap ==========
tng_df = (
  tng.filter(col("FromDate").isNotNull() & col("ToDate").isNotNull() & col("SectionBreakStartKM").isNotNull())
     .withColumn("Tng_From_d", F.coalesce(F.to_date("FromDate","dd/MM/yyyy"), F.to_date("FromDate","yyyy-MM-dd")))
     .withColumn("Tng_To_d",   F.coalesce(F.to_date("ToDate","dd/MM/yyyy"),   F.to_date("ToDate","yyyy-MM-dd")))
     .withColumn("Tng_From_ts", F.unix_timestamp(F.coalesce(F.to_timestamp("FromDate","dd/MM/yyyy"),
                                                            F.to_timestamp("FromDate","yyyy-MM-dd"))))
     .withColumn("Tng_To_ts",   F.unix_timestamp(F.coalesce(F.to_timestamp("ToDate","dd/MM/yyyy"),
                                                            F.to_timestamp("ToDate","yyyy-MM-dd"))))
     .select(
        F.col("BaseCode").alias("Tng_BaseCode"),
        F.col("SectionBreakStartKM").cast("double").alias("Tng_SectionBreakStartKM"),
        F.col("SectionBreakFinishKM").cast("double").alias("Tng_SectionBreakFinishKM"),
        "Tng_From_d","Tng_To_d","Tng_From_ts","Tng_To_ts","Tonnage"
     )
)

# Since Tonnage is usually sparser, first apply rough filtering by BaseCode and date, then check interval overlap
joined_for_tng = (
  joined
  .withColumn("Tc_r_date_ts", F.unix_timestamp(F.coalesce(F.to_timestamp("Tc_r_date","dd/MM/yyyy"),F.to_timestamp("Tc_r_date","yyyy-MM-dd"))))
  .withColumn("join_Base", col("Tc_BaseCode"))
)

cond_tng = (
   (col("join_Base") == col("Tng_BaseCode")) &
   (col("Tc_r_date_ts") >= col("Tng_From_ts")) &
   (col("Tc_r_date_ts") <= col("Tng_To_ts")) &
   (col("Wagon_SectionBreakStartKM") == col("Tng_SectionBreakStartKM")) &
   (col("Wagon_SectionBreakFinishKM")  == col("Tng_SectionBreakFinishKM"))
)

joined_all = joined_for_tng.join(tng_df, cond_tng, how="left")

df_result = (
  joined_all
  .groupBy(
      "Tc_BaseCode", 
      "Tc_BaseCode_Mapped",
      "Tc_SectionBreakStartKM",
      "Tc_r_date",
      "Wagon_RecordingDate",
      "Tc_p_key", 
    )
    .agg(
      F.avg("Wagon_Twist14m").alias("Wagon_Twist14m"),
      F.avg("Wagon_BounceFrt").alias("Wagon_BounceFrt"),
      F.avg("Wagon_BounceRr").alias("Wagon_BounceRr"),
      F.avg("Wagon_BodyRockFrt").alias("Wagon_BodyRockFrt"),
      F.avg("Wagon_BodyRockRr").alias("Wagon_BodyRockRr"),
      F.avg("Wagon_LP1").alias("Wagon_LP1"),
      F.avg("Wagon_LP2").alias("Wagon_LP2"),
      F.avg("Wagon_LP3").alias("Wagon_LP3"),
      F.avg("Wagon_LP4").alias("Wagon_LP4"),
      F.avg("Wagon_Speed").alias("Wagon_Speed"),
      F.avg("Wagon_BrakeCylinder").alias("Wagon_BrakeCylinder"),
      F.avg("Wagon_IntrainForce").alias("Wagon_IntrainForce"),
      F.avg("Wagon_Acc1").alias("Wagon_Acc1"),
      F.avg("Wagon_Acc2").alias("Wagon_Acc2"),
      F.avg("Wagon_Acc3").alias("Wagon_Acc3"),
      F.avg("Wagon_Acc4").alias("Wagon_Acc4"),
      F.avg("Wagon_Twist2m").alias("Wagon_Twist2m"),
      F.avg("Wagon_Acc1_RMS").alias("Wagon_Acc1_RMS"),
      F.avg("Wagon_Acc2_RMS").alias("Wagon_Acc2_RMS"),
      F.avg("Wagon_Acc3_RMS").alias("Wagon_Acc3_RMS"),
      F.avg("Wagon_Acc4_RMS").alias("Wagon_Acc4_RMS"),
      F.avg("Wagon_Rail_Pro_L").alias("Wagon_Rail_Pro_L"),
      F.avg("Wagon_Rail_Pro_R").alias("Wagon_Rail_Pro_R"),
      F.avg("Wagon_SND").alias("Wagon_SND"),
      F.avg("Wagon_VACC").alias("Wagon_VACC"),
      F.avg("Wagon_VACC_L").alias("Wagon_VACC_L"),
      F.avg("Wagon_VACC_R").alias("Wagon_VACC_R"),
      F.avg("Wagon_Curvature").alias("Wagon_Curvature"),
      F.avg("Wagon_Track_Offset").alias("Wagon_Track_Offset"),
      F.avg("Wagon_ICWVehicle").alias("Wagon_ICWVehicle"),
      F.avg("Wagon_SND_L").alias("Wagon_SND_L"),
      F.avg("Wagon_SND_R").alias("Wagon_SND_R"),
      F.sum("w_rows_in_bin").alias("w_row_count"),
      F.avg("Tonnage").alias("Tng_Tonnage")
    )    
)

df_result = df_result.select(
    "Tc_BaseCode", "Tc_BaseCode_Mapped",
    "Tc_SectionBreakStartKM",
    "Tc_r_date",
    "Wagon_RecordingDate","Tc_p_key", "Wagon_Twist14m",	"Wagon_BounceFrt","Wagon_BounceRr","Wagon_BodyRockFrt",	"Wagon_BodyRockRr","Wagon_LP1","Wagon_LP2","Wagon_LP3","Wagon_LP4","Wagon_Speed","Wagon_BrakeCylinder",	"Wagon_IntrainForce","Wagon_Acc1","Wagon_Acc2","Wagon_Acc3","Wagon_Acc4","Wagon_Twist2m","Wagon_Acc1_RMS",	"Wagon_Acc2_RMS","Wagon_Acc3_RMS","Wagon_Acc4_RMS","Wagon_Rail_Pro_L","Wagon_Rail_Pro_R","Wagon_SND",	"Wagon_VACC","Wagon_VACC_L","Wagon_VACC_R","Wagon_Curvature","Wagon_Track_Offset","Wagon_ICWVehicle",	"Wagon_SND_L","Wagon_SND_R","w_row_count","Tng_Tonnage"
)



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

# MAGIC %sql
# MAGIC -- SELECT count(*) FROM dev_adlunise.predictive_maintenance_uofa_2025.testcontext;
# MAGIC
# MAGIC SELECT * FROM dev_adlunise.predictive_maintenance_uofa_2025.testcontext limit 10
# MAGIC
# MAGIC -- SELECT * FROM dev_adlunise.predictive_maintenance_uofa_2025.trainingcontext limit 10
