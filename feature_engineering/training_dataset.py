# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # training_dataset
# MAGIC
# MAGIC \<Enrichment description\>.
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
# MAGIC | 2025-08-26 | Zi Lun Ma | Initial version. |

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
# MAGIC Ensure that the dataset resulting from executing the enrichment is stored in a Spark dataframe called 'df_result' e.g.  
# MAGIC `df_result = spark.sql("""`  
# MAGIC `  ...your SQL Code here`  
# MAGIC `""")` 
# MAGIC <br/><br/>
# MAGIC
# MAGIC ### Returning metrics to record against the Task Run
# MAGIC If you would like to return one of more metrics regarding the running of the code in this notebook, simply declare a variable 'run_output' and populate it with a valid JSON string containing your metrics.  At the end of the execution of this notebook, the value of the run_output will be recorded against the Task Run record.  For example, to record the version number of the model that is used to run inference, you might do something like:  
# MAGIC `run_output = '{ "model_version_number": 5 }'`
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

# MAGIC %sql
# MAGIC WITH
# MAGIC -- Wagon features (30-day window)
# MAGIC wagon_features AS (
# MAGIC     SELECT
# MAGIC         tc.p_key,
# MAGIC         tc.BaseCode,
# MAGIC         tc.SectionBreakStartKM,
# MAGIC         tc.r_date,
# MAGIC         AVG(w.Speed) AS avg_speed,
# MAGIC         MIN(w.Speed) AS min_speed,
# MAGIC         MAX(w.Speed) AS max_speed,
# MAGIC         MAX(w.Twist14m) AS max_twist14m,
# MAGIC         COUNT(DISTINCT w.RecordingDate) AS window_count
# MAGIC     FROM dev_adlunise.predictive_maintenance_uofa_2025.trainingcontext tc
# MAGIC     LEFT JOIN dev_adlunise.predictive_maintenance_uofa_2025.wagondata w
# MAGIC       ON tc.BaseCode = w.BaseCode
# MAGIC      AND tc.SectionBreakStartKM = w.SectionBreakStartKM
# MAGIC      AND w.RecordingDate BETWEEN date_sub(tc.r_date, 30) AND tc.r_date
# MAGIC     GROUP BY tc.p_key, tc.BaseCode, tc.SectionBreakStartKM, tc.r_date
# MAGIC ),
# MAGIC
# MAGIC -- Tonnage features (cumulative + up to r_date)
# MAGIC tonnage_features AS (
# MAGIC     SELECT
# MAGIC         tc.p_key,
# MAGIC         SUM(t.Tonnage) AS cumu_tonnage,
# MAGIC         SUM(
# MAGIC           CASE 
# MAGIC             WHEN tc.r_date BETWEEN to_date(t.FromDate, 'd/M/y') AND to_date(t.ToDate, 'd/M/y')
# MAGIC               THEN (DATEDIFF(tc.r_date, to_date(t.FromDate, 'd/M/y')) * 1.0 /
# MAGIC                    NULLIF(DATEDIFF(to_date(t.ToDate, 'd/M/y'), to_date(t.FromDate, 'd/M/y')),0)
# MAGIC                   ) * t.Tonnage
# MAGIC             WHEN to_date(t.ToDate, 'd/M/y') < tc.r_date 
# MAGIC               THEN t.Tonnage
# MAGIC             ELSE 0
# MAGIC           END
# MAGIC         ) AS tonnage_to_rdate
# MAGIC     FROM dev_adlunise.predictive_maintenance_uofa_2025.trainingcontext tc
# MAGIC     LEFT JOIN `09ad024f-822f-48e4-9d9e-b5e03c1839a2`.predictive_maintenance_uofa_2025.tonnagedata t
# MAGIC       ON tc.BaseCode = t.BaseCode
# MAGIC      AND tc.SectionBreakStartKM = t.SectionBreakStartKM
# MAGIC     GROUP BY tc.p_key
# MAGIC ),
# MAGIC
# MAGIC training_dataset AS (
# MAGIC     SELECT
# MAGIC         tc.p_key,
# MAGIC         tc.BaseCode,
# MAGIC         tc.SectionBreakStartKM,
# MAGIC         tc.r_date,
# MAGIC         DATEDIFF(
# MAGIC             day,
# MAGIC             last_fail_if_available_otherwise_null,
# MAGIC             break_date
# MAGIC         ) AS days_since_last_failure,
# MAGIC         -- Wagon features
# MAGIC         wf.avg_speed,
# MAGIC         wf.max_twist14m,
# MAGIC         wf.window_count,
# MAGIC         -- Tonnage features
# MAGIC         tf.cumu_tonnage,
# MAGIC         tf.tonnage_to_rdate,
# MAGIC         -- Label
# MAGIC         tc.target,
# MAGIC         tc.rul
# MAGIC     FROM dev_adlunise.predictive_maintenance_uofa_2025.trainingcontext tc
# MAGIC     LEFT JOIN wagon_features wf ON tc.p_key = wf.p_key
# MAGIC     LEFT JOIN tonnage_features tf ON tc.p_key = tf.p_key
# MAGIC )
# MAGIC
# MAGIC SELECT * 
# MAGIC FROM training_dataset;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Notebook End
# MAGIC
# MAGIC **(Do not modify/delete the following cell)**
# MAGIC
# MAGIC ####Important: 
# MAGIC Ensure that the result of your enrichment is stored in a Spark dataframe called 'df_result'. For example:  
# MAGIC `df_result = spark.sql("""`  
# MAGIC `  ...your SQL Code here`  
# MAGIC `""")`
# MAGIC

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
