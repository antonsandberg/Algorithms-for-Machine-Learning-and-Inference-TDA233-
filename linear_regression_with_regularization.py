import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt


def fit_linear_with_regularization(X, y, alpha):
    N = len(y)
    # w = (X'X + N*aI)^(-1)(X'y)
    # Chose to use with N here, as used in the lecture notes
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + N*np.identity(np.shape(X)[1])*alpha),
                  np.matmul(np.transpose(X), y))
    return w


def predict(X, w):
    # Use the calculated weight matrix for the new data
    y_prediction = np.matmul(X, w)
    return y_prediction



def plot_training(X_train, y_train):
    # Just for visualizing the data for our own sanity
    plt.scatter(X_train[:, 1], y_train, label='Real data')
    plt.title('Visualization of the training data')
    plt.xlabel('First attribute')
    plt.ylabel('Diabetes data')
    plt.legend()
    plt.show()


def plot_prediction(X_test, y_test, y_pred):
    # Plotting the test data against the prediction data
    plt.plot(X_test[:, 1], y_test, '*', label='Real data')
    plt.plot(X_test[:, 1], y_pred, '*', label='Prediction')
    plt.xlabel('First attribute')
    plt.ylabel('Diabetes data')
    plt.title('Diabetes test data against prediction data')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compute the mean squared error
    # mse = mean_squared_error(y_test, y_pred)
    mse = 1/len(y_test)*np.sum((y_test - y_pred)**2)
    return mse


# Load the diabetes dataset, X input, y output
X, y = datasets.load_diabetes(return_X_y=True)
X = np.array(X)

# Adding the first column in X (ones)
X = np.c_[np.ones(len(X)), X]
# Split the dataset into training and test set
num_test_elements = 20

X_train = X[:-num_test_elements]
X_test = X[-num_test_elements:]

y_train = y[:-num_test_elements]
y_test = y[-num_test_elements:]

# Set alpha
alpha = 0.00000001
# Train using linear regression with regularization and find optimal model
w = fit_linear_with_regularization(X_train, y_train, alpha)
# Make predictions using the testing set X_test
y_pred = predict(X_test, w)
error = plot_prediction(X_test, y_test, y_pred)
print(f'Mean Squared error is: {error}')
plot_training(X_train, y_train)


