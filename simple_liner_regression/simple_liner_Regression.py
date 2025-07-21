import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

print(y_pred)
print(y_test)

# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
#
# plt.scatter(X_test, y_test, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience test')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()