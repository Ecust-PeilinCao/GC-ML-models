import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from xgboost import XGBRegressor

df = pd.read_excel('feature matrix.xlsx')
scaler = MinMaxScaler()
data = df.iloc[:, 2:-1]
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
vis = pd.DataFrame(df.iloc[:, -1])
ILid = df.iloc[:, 1]
data = pd.concat([ILid, data, vis], axis=1)
data["Group"] = data["Group"].astype(int)
data["Experimental"] = data["Experimental"].astype(float)
data = data.drop("Group", axis=1)
X = data.drop('Experimental', axis=1)
Y = data['Experimental']


regressor = XGBRegressor(max_depth=10,
                         learning_rate=0.27,
                         n_estimators=200)
# the learning_rate for viscosity, refractive index, surface tension, and thermal conductivity are
# 0.27, 0.14, 0.12 and 0.16, respectively.
model = regressor.fit(X, Y)
train = model.predict(X)
train = pd.DataFrame(train)
print("result: ", r2_score(Y, train), np.sqrt(metrics.mean_squared_error(Y, train)))
