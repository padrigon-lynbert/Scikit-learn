import sklearn
from sklearn import datasets

dir(datasets)

iris = datasets.load_iris()

print(iris.data)
print(iris.target_names)
