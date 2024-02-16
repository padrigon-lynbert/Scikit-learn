import sklearn
from sklearn import datasets

dir(datasets)

iris = datasets.load_iris()

# print(iris.data)
# print(iris.target_names)
# print(iris.DESCR)

from sklearn.datasets import fetch_openml
mice = fetch_openml(name='miceprotein', version=4)
mice.details

print('shape: ', mice.data.shape)

print("missing values:", mice.data.isnull().sum().sum)

import seaborn as sns
import matplotlib.pyplot as plt

subset_features = mice.data.iloc[:, :5]
sns.pairplot(subset_features) 
plt.show()