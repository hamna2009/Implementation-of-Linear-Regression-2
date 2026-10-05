import matplotlib.pyplot as visual
import numpy
import sklearn
import pandas
from sklearn.metrics import mean_squared_error, r2_score

#dataset 
ds = pandas.read_csv('xydataset.csv', header=None, usecols=[1, 2], names=['X', 'Y'])

print(ds.describe())
print(ds.corr())
visual.scatter(ds['X'], ds['Y'])
visual.show()

X = ds[['X']].values   # double brackets keep it 2D, shape (100, 1)
Y = ds['Y'].values     # 1D, shape (100,)

X_train = X[:-20]
X_test = X[-20:]

Y_train = Y[:-20]
Y_test = Y[-20:]

regr = sklearn.linear_model.LinearRegression()
regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

print("Coefficient:", regr.coef_)
print("Intercept:", regr.intercept_)
print("MSE: %.2f" % mean_squared_error(Y_test, Y_pred))
print("RMSE: %.2f" % numpy.sqrt(mean_squared_error(Y_test, Y_pred)))
print("R2: %.2f" % r2_score(Y_test, Y_pred))

visual.scatter(X_test, Y_test, color="grey", label="Actual (test)")
visual.plot(X_test, Y_pred, color="blue", linewidth=3, label="Predicted")
visual.xlabel("x")
visual.ylabel("y")
visual.legend()
visual.show()
