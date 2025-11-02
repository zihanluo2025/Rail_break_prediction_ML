# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Feature engineering thresholds
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
else:
    # default to training
    base_table = "`09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.preprocess_training_table"

df_result = spark.sql(

f"""
WITH training AS (
  SELECT *, 
  YEAR(Tc_r_date) AS Year_Partition, 
  MONTH(Tc_r_date) AS Month_Partition
  --FROM `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.preprocess_training_table
  FROM {base_table}
),
-- Monthly sensor aggregation per each rail section
thresholds AS (
  SELECT 
    Tc_BaseCode, 
    Tc_SectionBreakStartKM,
    Year_Partition,
    Month_Partition,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(Wagon_Twist14m)) AS Threshold_Twist14m,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(Wagon_Twist2m)) AS Threshold_Twist2m,

    COALESCE(AVG(ABS(Wagon_BounceFrt)) + 2 * STDDEV(ABS(Wagon_BounceFrt)), 0) AS Threshold_BounceFrt,
    COALESCE(AVG(ABS(Wagon_BounceRr)) + 2 * STDDEV(ABS(Wagon_BounceRr)), 0) AS Threshold_BounceRr,

    COALESCE(AVG(ABS(Wagon_BodyRockFrt)) + 2 * STDDEV(ABS(Wagon_BodyRockFrt)), 0) AS Threshold_BodyRockFrt,
    COALESCE(AVG(ABS(Wagon_BodyRockRr)) + 2 * STDDEV(ABS(Wagon_BodyRockRr)), 0) AS Threshold_BodyRockRr,

    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_LP1) AS Threshold_LP1,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_LP2) AS Threshold_LP2,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_LP3) AS Threshold_LP3,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_LP4) AS Threshold_LP4,

    COALESCE(AVG(Wagon_Speed) + 2 * STDDEV(Wagon_Speed), 0) AS Threshold_Speed,

    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_BrakeCylinder) AS Threshold_BrakeCylinder,

    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_BodyRockRr) AS Threshold_IntrainForce,

    COALESCE(AVG(Wagon_Acc1) + 2 * STDDEV(Wagon_Acc1), 0) AS Threshold_Acc1,
    COALESCE(AVG(Wagon_Acc2) + 2 * STDDEV(Wagon_Acc2), 0) AS Threshold_Acc2,
    COALESCE(AVG(Wagon_Acc3) + 2 * STDDEV(Wagon_Acc3), 0) AS Threshold_Acc3,
    COALESCE(AVG(Wagon_Acc4) + 2 * STDDEV(Wagon_Acc4), 0) AS Threshold_Acc4,

    COALESCE(AVG(ABS(Wagon_Acc1_RMS)) + 2 * STDDEV(ABS(Wagon_Acc1_RMS)), 0) AS Threshold_Acc1_RMS,
    COALESCE(AVG(ABS(Wagon_Acc2_RMS)) + 2 * STDDEV(ABS(Wagon_Acc2_RMS)), 0) AS Threshold_Acc2_RMS,
    COALESCE(AVG(ABS(Wagon_Acc3_RMS)) + 2 * STDDEV(ABS(Wagon_Acc3_RMS)), 0) AS Threshold_Acc3_RMS,
    COALESCE(AVG(ABS(Wagon_Acc4_RMS)) + 2 * STDDEV(ABS(Wagon_Acc4_RMS)), 0) AS Threshold_Acc4_RMS,

    COALESCE(AVG(ABS(Wagon_Rail_Pro_L - Wagon_Rail_Pro_R)) + 2 * STDDEV(ABS(Wagon_Rail_Pro_L - Wagon_Rail_Pro_R)), 0) AS Threshold_Rail_Pro,

    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(Wagon_SND)) AS Threshold_SND,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(Wagon_SND_L)) AS Threshold_SND_L,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(Wagon_SND_R)) AS Threshold_SND_R,

    COALESCE(AVG(Wagon_VACC) + 2 * STDDEV(Wagon_VACC), 0) AS Threshold_VACC,
    COALESCE(AVG(Wagon_VACC_L) + 2 * STDDEV(Wagon_VACC_L), 0) AS Threshold_VACC_L,
    COALESCE(AVG(Wagon_VACC_R) + 2 * STDDEV(Wagon_VACC_R), 0) AS Threshold_VACC_R,
    -- AVG(Wagon_Curvature) AS Threshold_Curvature,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Wagon_Track_Offset) AS Threshold_Track_Offset
FROM training
GROUP BY Tc_BaseCode, Tc_SectionBreakStartKM, Year_Partition, Month_Partition
ORDER BY Tc_BaseCode, Tc_SectionBreakStartKM, Year_Partition, Month_Partition)

SELECT * FROM thresholds
""")

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
