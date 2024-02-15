import sklearn
from sklearn.datasets import load_iris

load_iris()

x,y = load_iris(return_X_y=True)

from sklearn.linear_model import LinearRegression
Model = LinearRegression()

Model.fit(x,y)
# print(Model.predict(x))


from sklearn.neighbors import KNeighborsRegressor
mod = KNeighborsRegressor()

mod.fit(x,y)
# print(mod.predict(x))
from matplotlib import pyplot as plt

pred = mod.predict(x)
# plt.scatter(pred, y)


# plt.show()

import pandas as pd
from sklearn.datasets import fetch_openml

df = fetch_openml('titanic', version=1, as_frame=True)['data']

# print(df.info())
# print(df.isnull())
print(df.isnull().sum())

import seaborn as sns

sns.set()
miss_val_per = pd.DataFrame( (df.isnull().sum()/len(df)) * 100 )
miss_val_per.plot(kind='bar', title='Missing values(%)', ylabel='%')

plt.tight_layout()
# plt.savefig("miss_val_per.png")

print(f'size of dataset: {df.shape}') # size of dataset: (1309, 13)

df.drop(['body'], axis=1, inplace=True) #axis = 1(row) # size of dataset: (1309, 12) 
print(f'size of dataset: {df.shape}')

from sklearn.impute import SimpleImputer
print(f"# of null before imputing: {df.age.isnull().sum()}") # # of null before imputing: 263
imp = SimpleImputer(strategy='mean')

df['age'] = imp.fit_transform(df[['age']])
print(f"# of null after imputing: {df.age.isnull().sum()}") # # of null after imputing: 0

def get_param(df):
    parameters = {}
    for col in df.columns[df.isnull().any()]:

        if df[col].dtype == 'float64' or df[col].dtype == 'int64' or df[col].dtype == 'int32': strategy = 'mean'
        else: 
            stratedy = 'most_frequent'

            missing_values = df[col][df[col].isnull()].values[0]
            parameters[col] = {'missing_values: {missing_values}', 'strategy: {strategy}'}
    return parameters

print(get_param(df))

# plt.show()    