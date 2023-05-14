# Diabetes prediction Using Machine Learning
Machine Learning algorithms have been trained on a dataset of 768 patients, where each patient is described by 8 clinical features such as glucose level, blood pressure, BMI, etc. The objective is to build a model that can predict whether a patient has diabetes or not based on these features.

Dataset
The dataset used in this project is the Pima Indians Diabetes Dataset, which is available on the UCI Machine Learning Repository. It contains 768 instances, where each instance has 8 clinical features and a binary target variable indicating whether the patient has diabetes or not.

Machine Learning algorithms
Several Machine Learning algorithms have been used to build the prediction models, including:

Logistic Regression
K-Nearest Neighbors
Decision Trees
Random Forests
Support Vector Machines

Later,svm otherwise Support Vector Machines is taken as final as it performs comparitively well as cmpared to other models

File Description
diabetes.csv : The dataset used for training and testing the models.
Diabetes_Prediction.ipynb : A Jupyter Notebook containing the code used for data preprocessing, model training, and model evaluation.
README.md : This file.
trained_model.sav:To save the model trained
Requirements
The following Python packages are required to run the code in Diabetes_Prediction.ipynb:

pandas
numpy
matplotlib
seaborn
sklearn
streamlit

Usage
To run the web application, navigate to the directory containing Diabetes_Prediction.py and run the command in the terminal:
streamlit run Diabetes_Prediction.py
This will start the web application on a local server, and you can access it by opening the URL displayed in the terminal.

Conclusion
The results show that the Random Forest algorithm performs the best with an accuracy of 79% and an AUC score of 0.85. This project demonstrates how Machine Learning can be used to predict the likelihood of a patient having diabetes based on clinical features, which can help in early diagnosis and treatment of the disease.
