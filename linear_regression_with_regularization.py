import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt


def fit_linear_with_regularization(X, y, alpha):
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + alpha),
                  np.matmul(np.transpose(X), y))
    return w


def predict(X, w):
    y_prediction = np.matmul(X, w)
    return y_prediction


def plot_prediction(X_test, y_test, y_pred):
    plt.scatter(X_test[:, 0], y_test, label='Real data')
    plt.scatter(X_test[:, 0], y_pred, label='Prediction')
    plt.xlabel('First attribute')
    plt.ylabel('Diabetes data')
    plt.title('Linear regression with regularization')
    plt.legend()
    plt.show()

    # Compute the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    return mse


# Load the diabetes dataset, X input, y output
X, y = datasets.load_diabetes(return_X_y=True)

# Split the dataset into training and test set
num_test_elements = 20

X_train = X[:-num_test_elements]
X_test = X[-num_test_elements:]

y_train = y[:-num_test_elements]
y_test = y[-num_test_elements:]

# Set alpha
alpha = 0.01
# Train using linear regression with regularization and find optimal model
w = fit_linear_with_regularization(X_train, y_train, alpha)
# Make predictions using the testing set X_test
y_pred = predict(X_test, w)
error = plot_prediction(X_test, y_test, y_pred)

print(f'Mean Squared error is: {error}')
