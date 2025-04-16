##  Agriculture Yield Prediction

A machine learning project developed for the DeepData Hackathon to predict crop yields based on agricultural factors.

##  Problem Statement
Agricultural yield is influenced by a variety of factors including soil conditions, rainfall, temperature, and crop type. Accurate yield prediction enables farmers and agribusinesses to make informed decisions, optimize resource allocation, and improve productivity.
This project aims to build a predictive model that leverages real-world data to forecast agricultural yield in a smart and scalable way.

## Project structure
Agriculture-yield-prediction/
├── data/
│   └── crop_yield_train.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   └── data_handling.py
│   └── visualization.py
│   └── model_training.py
├── results/
│   └── plots/
│   └── metrics.txt
├── requirements.txt
├── README.md
└── .gitignore

##  Dataset
- Provided by organizers.
- Combination of raw & preprocessed features.
- Handled missing values, outliers, encoding, and scaling.

##  Tech Stack
- Language: Python 3.x
- Libraries: Dask, Pandas, NumPy, joblib , Matplotlib, Seaborn, Scikit-Learn, XGBoost, SHAP
- Tools: Google Colab, GitHub
  
##  Workflow

1️⃣ Data Cleaning (missing values, encoding, scaling)  
2️⃣ Exploratory Data Analysis (EDA)  
3️⃣ Model Training: RandomForest, XGBoost
4️⃣ Evaluation: F1, AUC-ROC, Accuracy  
5️⃣ Explainability: SHAP and Feature Importance  


