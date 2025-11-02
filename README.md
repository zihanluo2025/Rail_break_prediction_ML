# RAIL-PG-2 
## Software Engineering & Project (COMP SCI 7015)

| Name | Student ID | . |  Name| Student ID | . |  Name| Student ID | 
| :-- | :-- |  - |  :--|  :--|   - |  :--|  :--|       
| Tao Xu | a1937511 | . | Jinchao Yuan| a1936476 | . | Zilun Ma |  a1915860|
|  Di Zhu | a1919727 |  . |Xin Wei| a1912958| .| Yifan Gu | a1909803|
|  Tianhua Zhang | a1915934 |  . |Zihan Luo| a1916700| .| Sheng Wang | a1903948|


# Abstract
Railway track health Prediction Project Based on Databricks and Delta tables. We will integrate data such as train load, on-board sensors (ICW), and track segment identification to train a model to predict whether track breakage will occur within the next 30 days and provide a reference for the remaining useful life (RUL).

## Project Structure
```
.
├─ EDA_analyze/              # Exploratory Data Analysis (EDA): data overview, visualizations
├─ feature_engineering/      # Feature engineering: cleaning, scaling, resampling, windows/aggregations
├─ feature_selection/        # Feature selection: filter/wrapper/embedded (e.g., Group Lasso)
├─ machine_learning/         # ML training & evaluation: baselines, configs, metrics, submissions
├─ docs/                     # Project documentation (see expanded tree below)
│  ├─ agendas_minutes/         # Meeting agendas & public minutes (team/client-facing)
│  ├─ assignments/             # Course deliverables
│  ├─ minutes(internal)/       # Internal minutes: sensitive notes, risks, decisions
│  ├─ research/                # Background research: papers, notes, experiment ideas, summaries
│  └─ snapshots/               # Course deliverables snapshots: weekly status
├─ .gitignore                # Ignore rules
└─ README.md                 # You are here
```
