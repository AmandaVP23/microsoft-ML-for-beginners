import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

# Load the diabetes dataset
X, y = datasets.load_diabetes(return_X_y=True);

# Print the shape of the data and the first row
print(X.shape)
print(X[0])

X = X[:, 2]
X = X.reshape(-1, 1)
print(X.shape)

# Split the data into training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

print("After split")
print(X_train.shape)
print(X_test.shape)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred.shape)

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Scaled BMIs')
plt.ylabel('Disease Progression')
plt.title('A graph plot showing diabetes progression against BMI')

plt.show()
