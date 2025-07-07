![UTA-DataScience-Logo](https://github.com/user-attachments/assets/6d626bcc-5430-4356-927b-97764939109d)

# Airline-Passenger-Satisfaction
* This repository presents a complete machine learning pipeline to classify airline passenger satisfaction levels using the Airline Passenger Satisfaction dataset, including data cleaning, feature engineering, model training, evaluation, and comparison.
* Kaggle Challenge Link: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

## Overview

  * **Definition of the tasks / challenge**: The task is to predict whether a passenger is satisfied with their flight experience based on various numerical and categorical features (e.g., flight distance, service ratings, delays). I formulated this as a binary classification problem.
  * **My approach**: The approach involved extensive preprocessing (handling missing values, outliers, and encoding categorical features), feature scaling, and applying both baseline models (Logistic Regression, KNN) and advanced classifiers (Random Forest, XGBoost). Hyperparameter tuning via GridSearchCV was used to improve performance.
  * **Summary of the performance achieved**: The best-performing models—**Random Forest and XGBoost**—achieved up to **94% accuracy** on both internal and external test sets, with strong F1-scores and AUC values, highlighting their effectiveness in this setting.

## Summary of Workdone

### Data

* Data:
  * **Type**: Tabular Dataset
    * Input: Train and Test CSV file with 25 features including flight and service metrics
    * Target: Binary satisfaction label (**0** = neutral/dissatisfied, **1** = satisfied)
  * **Size**:
    * Training set: 103,904 rows and 25 columns
    * Test set: 25,976 rows and 25 columns
      
#### Data Cleaning

* Handle missing values
    * **Numerical columns**:
       * Filled missing values in **Arrival Delay in Minutes** with **median**
    * **Categorical columns**:
       * Dataset did not contain missing values in categorical columns, no imputation needed

* Handle outliers
    * **Numerical Columns Processesed**:
       * Age, Flight Distance, Arrival Delay in Minutes, Departure Delay in Minutes
    * Replace outliers using **upper and lower bounds** computed from IQR
    * Visualized impact using **box-plots** before and after handling

#### Preprocessing

* **Derived Features**
  * Create new feature:
    * Delay Difference = Arrival Delay in Minutes - Departure Delay in Minutes
      
* **Feature Removal**
  * Dropped highly correlated features:
    * Inflight wifi service and Arrival Delay in Minutes

* **Feature Scaling and Encoding**
  * Numerical columns:
    * Applied **StandardScaler** to normalize all numeric features for regression models
   
  * Categorical columns:
    * Applied **OneHotEncoder** for categorical features (e.g., Gender, Customer Type, Type of Travel, etc.)

#### Data Visualization

**Satisfaction Distribution Pie Chart**
* The dataset is **slightly imbalanced**
  * 56.7% of passengers classified as **"Neutral or Dissatisfied"**
  * 43.3% classified as **"Satisfied"**
![image](https://github.com/user-attachments/assets/3393b2d3-ab7a-424b-bdf8-8f127b222f90)

**Flight Distance vs Type of Travel by Class**
* In Business Class, passengers on business travel tend to take longer flights and are more likely to be satisfied.

* Eco and Eco Plus show relatively balanced satisfaction levels for both travel types, but with shorter flight distances overall.
![image](https://github.com/user-attachments/assets/d84fb2b4-b578-4437-9005-cef13ea79b9d)

**Arrival vs Departure Delay Scatterplot (by Satisfaction)**
![image](https://github.com/user-attachments/assets/f98c0f33-6b5a-4736-8a29-ad690346861c)
* This plot shows a positive correlation between arrival and departure delays
* Most delays are clustered below 400 minutes, with extreme outliers up to 1500+ minutes.
* Dissatisfied passengers (class 0) are more frequent in high-delay zones, suggesting delay could contribute to dissatisfaction.
  
### Problem Formulation

  * **Input**
    * Scaled numerical features and Encoded categorical features for baseline regression models
      
    * Original numerical features and Encoded categorical features for advanced tree-based models
      
  * **Output**
    * Binary classification (satisfaction)
      
  * **Models Used**
    * **Logistic Regression**
      * **Parameters**: C, penalty, solver, max_iter

    * **KNN**
      * **Parameters**: n_neighbors, weights, p
        
    * **Random Forest**
      * **Parameters**: n_estimators, max_depth, min_samples_split, min_samples leaf, max_features, bootstrap, random_state

    * **XGBoost**
      * **Parameters**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, eval_metric, random_state
     
  * **Loss/Optimization**
    * GridSearchCV for tuning hyperparameters

  * **Evaluation Metrics**
    * Confusion Metrics
    * Classification Report (Accuracy, Precision, Recall, F1-Score)
    * ROC-AUC Curve
    * Use Test.csv as external test set to compare with internal test set (train split)
     
### Training
* **Environment**
  * Google Colab (Python, Scikit-learn, XGBoost)
  * All models trained using 80-20 split with 3-fold cross-validation during tuning
  * Training was efficient due to model simplicity and modest dataset size

### Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | AUC  |
| ------------------- | -------- | --------- | ------ | -------- | ---- |
| Logistic Regression | 87%      | 0.87      | 0.82   | 0.84     | 0.92 |
| KNN                 | 92%      | 0.93      | 0.87   | 0.90     | 0.97 |
| Random Forest       | 94%      | 0.94      | 0.92   | 0.93     | 0.99 |
| XGBoost             | 94%      | 0.95      | 0.92   | 0.93     | 0.99 |

![image](https://github.com/user-attachments/assets/dfcce4b0-8088-4cd6-bd89-af64f210bb19) 
![image](https://github.com/user-attachments/assets/7250f670-5880-4c6f-a5da-ce807a770918)
![image](https://github.com/user-attachments/assets/c90829a4-234f-46e8-a182-c9c2ccfe41e8)

### Conclusions
* XGBoost and Random Forest significantly outperformed simpler models like KNN and Logistic Regression
* Data Cleaning, Feature engineering, Data Preprocessing played a critical role
* Ratings for services like Online Boarding, Type of Travel, Seat Class were among the most influential features

### Future Work
* Try ensemble stacking or boosting with meta-models
* Integrate additional external datasets (e.g., weather or route delays)
* Experiment with deep learning models (e.g., neural networks)

## How to reproduce results
* **Setup**
  * Recommend: Google Colab, Jupyter Notebook, VSCode
  * The code runs efficiently on standard CPU hardware

* **Steps**
  1. Download dataset from Kaggle (https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
  2. Unzip and place in working directory
  3. Run the **airline_passenger_satisfaction.ipynb** to:
     * Preprocess data
     * View visualizations
     * Train and evaluate models
       
### Software Setup
* **Required packages**
  * Pandas, NumPy, Scikit-learn, XGBoost, matplotlib, seaborn

* **Installing packages in Jupyter/Colab**
  * ```sh
    !pip install pandas
    !pip install numpy
    !pip install scikit-learn
    !pip install xgboost
    !pip install matplotlib
    !pip install seaborn
    ``` 






