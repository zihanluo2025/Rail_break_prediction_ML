# RAIL-PG-2 
## Software Engineering & Project (COMP SCI 7015)

| Name | Student ID | . |  Name| Student ID | . |  Name| Student ID | 
| :-- | :-- |  - |  :--|  :--|   - |  :--|  :--|       
| Tao Xu | a1937511 | . | Jinchao Yuan| a1936476 | . | Zilun Ma |  a1915860|
|  Di Zhu | a1919727 |  . |Xin Wei| a1912958| .| Yifan Gu | a1909803|
|  Tianhua Zhang | a1915934 |  . |Zihan Luo| a1916700| .| Sheng Wang | a1903948|

## Scrum Master
- sprint 1: Zilun Ma
- sprint 2: Tianhua Zhang
- sprint 3: Di Zhu
- sprint 4: Jinchao Yuan
- sprint 5: Zihan Luo

## Achievements
TOP 2 
<img width="2880" height="1556" alt="36e4c77411164b9939482ca51aaae81f" src="https://github.com/user-attachments/assets/c06ab839-6b39-4bad-a04b-008b3455910d" />
<img width="2880" height="1556" alt="9aa85ffc03cfdd1d6239b5b7faad9e9e" src="https://github.com/user-attachments/assets/f87ae4b8-1508-4470-b297-5fb70c4f97f8" />
<img width="2880" height="1556" alt="dc4d5c557e73fe738bb17e6e4b518708" src="https://github.com/user-attachments/assets/074d742a-6a14-43f2-bf8b-2241b782c727" />


## Development environment
- Microsoft Azure Databricks
- Python
- Insight Factory platflom

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
