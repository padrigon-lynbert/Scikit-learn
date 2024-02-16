import sklearn
import pandas as pd

total_data = pd.read_csv('Building SVC Model/Seed_Data.csv')

# total_data.describe
# total_data.shape
x = total_data.iloc[:,0:7]
# x.info
y = total_data.iloc[:,7]
# y.describe

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=13)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
clf = svm.SVC()
clf.fit(x_train, y_train)

pred_clf = clf.predict(x_test)

print(sklearn.metrics.accuracy_score(y_test,pred_clf))
print(sklearn.metrics.classification_report(y_test,pred_clf))