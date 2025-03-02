# Diabetes Prediction using KNN

This project implements a K-Nearest Neighbors (KNN) classifier to predict whether a patient has diabetes based on certain diagnostic measurements.

## Project Overview

The goal of this project is to build a machine learning model that can accurately predict diabetes in patients using the Pima Indians Diabetes Dataset. The model uses the K-Nearest Neighbors algorithm, which classifies new data points based on the majority class of their k nearest neighbors in the feature space.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database, which includes the following features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0 or 1) indicating whether the patient has diabetes

## Implementation

The implementation includes the following steps:

1. **Data Loading and Preprocessing**:
   - Load the diabetes dataset
   - Separate features (X) and the target variable (y)
   - Apply normalization to scale all features to the range [0, 1]

2. **Model Training**:
   - Split the data into training and testing sets
   - Train a KNN classifier on the training data
   - Find the optimal value of k (number of neighbors) by testing different values

3. **Model Evaluation**:
   - Evaluate the model's performance on the test data
   - Calculate accuracy for different values of k

4. **Making Predictions**:
   - Use the trained model to predict diabetes for new patient data
   - Ensure proper normalization of new data before prediction

## Feature Importance

The model considers all features for prediction, with glucose levels and BMI being particularly important indicators for diabetes. The KNN algorithm implicitly weighs features based on their distances in the feature space.

## Results

The KNN model achieves approximately 83% accuracy with k=3, making it a reliable tool for diabetes prediction based on the given features.

## Usage

To use this model for predicting diabetes:

1. Ensure you have the required libraries installed:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```

2. Load your data and prepare it similarly to the training data

3. Use the following code to make predictions for new patients:
   ```python
   # Prepare new patient data (must have the same features as training data)
   new_data = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=feature_names)
   
   # Transform data using the same scaler
   new_data_normalized = sc.transform(new_data)
   
   # Convert back to DataFrame with column names
   new_data_normalized_df = pd.DataFrame(new_data_normalized, columns=feature_names)
   
   # Make prediction
   prediction = knn.predict(new_data_normalized_df)
   ```

## Future Improvements

Potential improvements to this project include:
- Feature selection to identify the most relevant features
- Comparing KNN with other classification algorithms
- Implementing cross-validation for more robust evaluation
- Handling class imbalance in the dataset

## References

- UCI Machine Learning Repository: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Scikit-learn documentation: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
