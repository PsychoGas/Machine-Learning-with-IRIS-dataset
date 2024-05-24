from pandas import read_csv
# Load dataset

url = "https://raw.githubusercontent.com/PsychoGas/ml_project/master/dataset.csv?token=GHSAT0AAAAAACSV6ROGLEAPOGAUS7SYRME2ZSQ2ZIA"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# Print number of rows and columns in datatset - shape
print(dataset.shape)


# Peek at the data - head
print(dataset.head(20))


# Checking number of rows related to a specific class - groupby
print(dataset.groupby('class').size())
