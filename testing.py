# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing the models 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


...
# Load dataset
url = "https://raw.githubusercontent.com/PsychoGas/ml_project/master/dataset.csv?token=GHSAT0AAAAAACSV6ROGLEAPOGAUS7SYRME2ZSQ2ZIA"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset -> load the dataset into a numpy array
array = dataset.values

# sepal length, sepal width, petal length, petal width will be one array X
X = array[:,0:4]

# y will have all the fifth column of all rows -> flower type -> result
y = array[:,4]


# We will use 80% of data to train and 20% to validate the model
"""
X_train: 80% of the feature data used for training the model.
X_validation: 20% of the feature data used for validating/testing the model.
Y_train: 80% of the target data used for training the model.
Y_validation: 20% of the target data used for validating/testing the model.
"""

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# We will use stratified 10 fold cross validation to train the model and test it
# Note that when we split the data, we used a random_state = 1 to make the split reproducible i.e the same split will be obtained if we run the code again hence the cross validation will be the same

# Accuracy -> ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100

"""
Algorithms:
Logistic Regression (LR): Finds a rule (line or curve) to separate two groups.
Linear Discriminant Analysis (LDA): Finds directions to separate multiple groups.
K-Nearest Neighbors (KNN): Looks at the nearest balls and decides based on their majority color.
Classification and Regression Trees (CART): Asks a series of questions to split the data into groups.
Gaussian Naive Bayes (NB): Uses probabilities to guess the color.
Support Vector Machines (SVM): Finds the best line or curve that separates the groups with the widest gap.
"""

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()