import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt


def fit_linear_with_regularization(X, y, alpha):
    N = len(y)
    # w = (X'X + N*aI)^(-1)(X'y)
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + N*np.identity(np.shape(X)[1])*alpha),
                  np.matmul(np.transpose(X), y))
    return w


def predict(X, w):
    y_prediction = np.matmul(X, w)
    return y_prediction


# Just for visualizing the data for our own sanity
def plot_training(X_train, y_train):

    plt.scatter(X_train[:, 1], y_train, label='Real data')
    plt.title('Visualization of the training data')
    plt.xlabel('First attribute')
    plt.ylabel('Diabetes data')
    plt.legend()
    plt.show()


def plot_prediction(X_test, y_test, y_pred):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Linear regression with regularization')
    ax1.plot(range(len(y_test)), y_test, label='Real data')
    ax1.plot(range(len(y_pred)), y_pred, label='Prediction')
    ax1.set(xlabel='Range(len)', ylabel='Diabetes data')
    ax1.set_title('With num data')
    ax1.legend()

    ax2.plot(X_test[:, 1], y_test, '*', label='Real data')
    ax2.plot(X_test[:, 1], y_pred, '*', label='Prediction')
    ax2.set(xlabel='First attribute', ylabel='Diabetes data')
    ax2.set_title('With diabetes data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Compute the mean squared error
    # mse = mean_squared_error(y_test, y_pred)
    mse = 1/len(y_test)*np.sum((y_test - y_pred)**2)
    return mse


# Load the diabetes dataset, X input, y output
X, y = datasets.load_diabetes(return_X_y=True)
X = np.array(X)
X = np.c_[np.ones(len(X)), X]
# Split the dataset into training and test set
num_test_elements = 20

X_train = X[:-num_test_elements]
X_test = X[-num_test_elements:]

y_train = y[:-num_test_elements]
y_test = y[-num_test_elements:]

# Set alpha
alpha = 0.00001
# Train using linear regression with regularization and find optimal model
w = fit_linear_with_regularization(X_train, y_train, alpha)
# Make predictions using the testing set X_test
y_pred = predict(X_test, w)
error = plot_prediction(X_test, y_test, y_pred)
print(f'Mean Squared error is: {error}')
plot_training(X_train, y_train)

