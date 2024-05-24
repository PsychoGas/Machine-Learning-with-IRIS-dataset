# Machine Learning with Iris Flower Dataset

This project demonstrates a basic machine learning workflow using the Iris Flower dataset. The goal is to train a model that can accurately predict the species of an iris flower based on its physical characteristics. The project includes data exploration, visualization, model comparison, and evaluation, ultimately selecting the Support Vector Machine (SVM) model for final predictions.

## Project Overview

The project consists of several stages, each implemented in separate Python scripts:

1. **Library Check** (`check_libraries.py`): Ensures all necessary libraries are installed and working.
2. **Data Exploration** (`view_data.py`): Loads and inspects the dataset, providing basic statistics and structure.
3. **Data Visualization** (`plots.py`): Visualizes the dataset using box plots, histograms, and scatter plots.
4. **Model Testing** (`testing.py`): Compares six different machine learning models to determine the best one.
5. **Model Training and Evaluation** (`main.py`): Trains the chosen model (SVM) and evaluates its performance.

## Installation
Ensure you have Python installed along with the necessary libraries. You can install the required libraries using the following command:

```bash
pip install pandas matplotlib scikit-learn
```

## Dataset

The Iris Flower dataset consists of 150 instances, each with four features: sepal length, sepal width, petal length, and petal width. The target variable is the species of the iris flower. More details about the dataset can be found [here](https://en.wikipedia.org/wiki/Iris_flower_data_set) (wikipedia page).

## Files and Scripts

1. **check_libraries.py**
    - Ensures all necessary libraries (pandas, matplotlib, scikit-learn) are installed and working properly.

2. **view_data.py**
    - Loads the Iris dataset, prints its shape, first 20 entries, and the number of rows related to each species.

3. **plots.py**
    - Visualizes the dataset using box plots, histograms, and scatter plots to identify potential patterns or algorithms.

4. **testing.py**
    - Compares six machine learning models (Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Tree, Naive Bayes, Support Vector Machine) using stratified 10-fold cross-validation to estimate accuracy. SVM showed the best performance.

5. **main.py**
    - Trains the SVM model using 80% of the dataset and evaluates its performance on the remaining 20%. Prints the accuracy, confusion matrix, and classification report, and identifies instances where predictions failed.

## Module Descriptions

- **pandas**: A powerful data manipulation and analysis library. It provides data structures like DataFrames to handle and process structured data.
- **matplotlib**: A plotting library used for creating static, animated, and interactive visualizations in Python.
- **scikit-learn**: A machine learning library that provides simple and efficient tools for data mining and data analysis. It includes implementations of various algorithms, including classification, regression, clustering, and more.

## Model Descriptions

- **Logistic Regression (LR)**: Finds a line or curve to separate two groups based on probabilities.
- **Linear Discriminant Analysis (LDA)**: Finds linear combinations of features that best separate multiple groups.
- **K-Nearest Neighbors (KNN)**: Classifies instances based on the majority class among the k-nearest neighbors.
- **Decision Tree (CART)**: Splits the data into groups based on a series of questions about the features.
- **Naive Bayes (NB)**: Uses Bayes' theorem and assumes independence between features to classify instances.
- **Support Vector Machine (SVM)**: Finds the optimal hyperplane that separates the data into different classes with the maximum margin.

## 10-Fold Cross Validation

10-fold cross-validation is a model evaluation method where the dataset is divided into 10 equal parts (folds). The model is trained on 9 parts and tested on the remaining part. This process is repeated 10 times, with each fold used once as the test set. The results are averaged to estimate the model's performance.

## Results

The Support Vector Machine (SVM) model provided the highest accuracy among the tested models. Below are the evaluation metrics for the SVM model:

- **Accuracy**: 96%
- **Confusion Matrix**: Displays the number of correct and incorrect predictions.
- **Classification Report**: Provides precision, recall, and F1-score for each class.

## Conclusion

The SVM model was selected for its superior performance in predicting the species of iris flowers. The project demonstrates a complete machine learning workflow, from data exploration and visualization to model selection and evaluation.

