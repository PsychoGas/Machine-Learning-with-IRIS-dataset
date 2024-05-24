# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
# Load dataset
url = "https://raw.githubusercontent.com/PsychoGas/ml_project/master/dataset.csv?token=GHSAT0AAAAAACSV6ROGLEAPOGAUS7SYRME2ZSQ2ZIA"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
