from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# Load dataset
url = "https://raw.githubusercontent.com/PsychoGas/Machine-Learning-with-IRIS-dataset/datasetUpload/dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Fit the dataset into model
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)  # Training the model with input n output
predictions = model.predict(X_validation)  # Making the model predict output using our tetsing data (20% of the dataset) and store the results in predictions

for i, j in (zip(predictions, Y_validation)):
    if (i != j):
        print("Error in predictions are: ")
        print("Predicted: ", i, "     -       Actual: ", j)

# Evaluate predictions score
print(accuracy_score(Y_validation, predictions) * 100, "%")
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
