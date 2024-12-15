[PLACEHOLDER]

# Stroke Risk Prediction: Medical Outcome Analysis
[View Notebook](https://github.com/TuringCollegeSubmissions/pjurci-DS.v2.5.3.2.5/blob/main/notebook.ipynb) | [Application](https://stroke-prediction-apgmtvsybdsgcvwzcfzsgs.streamlit.app/)

A machine learning analysis to predict stroke risk in patients using medical and lifestyle data. This study develops an ensemble model to identify high-risk patients and deploys an interactive prediction application.

## Overview

### Business Question 
How can we predict stroke risk in patients to enable early intervention and emergency preparedness?

### Key Findings
- Age, BMI, glucose level significant predictors
- Marriage status correlates with risk
- Model identifies 2/3 of stroke patients
- Ensemble methods improve prediction
- 5% stroke/95% non-stroke class imbalance

### Impact/Results
- Achieved 0.26 F1 score
- Deployed prediction application
- Created risk assessment tool
- Identified key risk factors
- Established intervention thresholds

## Data

### Source Information
- Dataset: Stroke Prediction Dataset
- Source: Kaggle (WHO-based)
- Size: ~5000 patients
- Year: 2021 or earlier
- Balance: 5% stroke cases

### Variables Analyzed
- Demographics
- Medical history
- Lifestyle factors
- Clinical measurements
- Risk indicators
- Outcome labels

## Methods

### Analysis Approach
1. Data Preprocessing
   - Imbalance handling
   - Feature engineering
   - Distribution analysis
2. Model Development
   - Multiple classifiers
   - Ensemble stacking
   - Performance optimization
3. Deployment
   - Streamlit application
   - Interactive interface
   - Risk calculation

### Tools Used
- Python (Data Science)
  - Pandas/Numpy: Data processing
  - Scikit-learn:
    - Logistic Regression
    - Random Forest
    - Stacking Classifier
    - Performance metrics
  - XGBoost: Gradient boosting
  - Imbalanced-learn: Class balancing
  - Matplotlib/Seaborn: Visualization
  - Joblib: Model persistence
  - Performance Metrics:
    - F1 Score (0.26)
    - Precision (0.17)
    - Recall (0.62)
- Streamlit: Application deployment

## Getting Started

### Prerequisites
```python
imbalanced_learn==0.12.3
ipython==8.12.3
joblib==1.4.2
matplotlib==3.8.4
numpy==2.2.0
pandas==2.2.3
scikit_learn==1.6.0
seaborn==0.13.2
shap==0.46.0
xgboost==2.1.3
streamlit==1.41.1
```

### Installation & Usage
```bash
git clone git@github.com:PJURC-data-science/stroke-prediciton.git
cd stroke-prediciton
pip install -r requirements.txt
jupyter notebook "Stroke Prediction.ipynb"
```

For the deployed model, open the [Streamlit Application](https://stroke-prediction-apgmtvsybdsgcvwzcfzsgs.streamlit.app/)

## Project Structure
```
stroke-prediction/
│   README.md
│   requirements.txt
│   Stroke Prediction.ipynb
|   utils.py
|   styles.css
|   app.py
|   stroke_prediction_model.joblib
└── data/
    └── healthcare-dataset-stroke-data.csv
```

## Strategic Recommendations
1. **Risk Assessment**
   - Set probability thresholds
   - Implement screening protocol
   - Monitor high-risk patients

2. **Model Application**
   - Use for initial screening
   - Recommend check-ups
   - Consider cost factors

3. **Clinical Integration**
   - Support doctor decisions
   - Enhance preventive care
   - Guide intervention timing

## Future Improvements
- Test additional models
- Test other ensemble methods (e.g., VotingClassifier with optimal weights)
- Expand grid search
- Test other methods for class imbalance (e.g., undersampling, oversampling)
- Update data recency