import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import math


def plot_err(x, y_train, y_test, lam_opt):
    plt.ylabel("ln(MSE)")
    plt.xlabel("ln(λ)")
    plt.title("Error for Ridge Regression")
    plt.plot(x, y_train, label="Training Error")
    plt.plot(x, y_test, label="Testing Error")
    plt.axvline(x=lam_opt, color='g', linestyle='--', 
                label="optimal lambda from 5-CV")
    plt.legend(loc="upper left")
    plt.savefig("lamda_err.png")
    plt.show()
    

def MSE(y_pred, y_truth):
    mse = 0
    for i in range(len(y_pred)):
        mse += ((y_truth[i] - y_pred[i]) ** 2)
    return math.log(mse / len(y_pred))
        

def ridge_regression(lam, X, y):
    # Finds the optimal weights using ridge regression i.e. least squares
    # with a regularization parameter defined by the l2-norm
    xtx = np.matmul(np.transpose(X), X)
    xtx_dim = np.shape(xtx)[0]
    w = (xtx + lam * np.identity(xtx_dim))
    w = np.linalg.inv(w)
    w = np.matmul(w, np.transpose(X))
    w = np.dot(w, y)
    return w


def error(k, pred, truth):
    total = 0
    for i in range(len(pred)):
        total += ((pred[i] - truth[i]) ** 2)
    return total


def cross_val(k, x_train, y, lams):
    fold_len = len(x_train) // k
    # Partition data into k-folds
    D, Y = [], []
    for i in range(0, len(x_train), fold_len):
        D.append(x_train[i : i + fold_len])
        Y.append(y[i : i + fold_len])
    D = D[:-1] # Got ride of the array of size 2 because bugs
    Y = Y[:-1]
    
    # Goes through every lambda and every subset
    all_cv = []
    for lam in lams:
        all_fitted = []
        for i in range(len(D)):
            test = D.pop(i)
            y_test = Y.pop(i)
            train = np.array([j for i in D for j in i])
            y_train = np.array([j for i in Y for j in i])
            D.insert(i, test)
            Y.insert(i, y_test)
            
            # Train the model with subset
            w = ridge_regression(lam, train, y_train)
            y_pred = np.dot(train, w)
            err = error(k, y_pred, y)
            all_fitted.append(err[0])
        
        # Compute average error for a specific lambda
        cv_err = (1 / k) * sum(all_fitted)
        all_cv.append((cv_err, lam))
            
    # return minimum cv error's lamda
    return min(all_cv, key = lambda t: t[0])
    

mat = scipy.io.loadmat('diabetes.mat')

# Get the data
x_train = np.array(mat["x_train"])
x_test = np.array(mat["x_test"])
y_train = np.array(mat["y_train"])
y_test = np.array(mat["y_test"])

# Test different lambdas to see which one is best ;)
lams = [.00001, .0001, .001, .01, .1, 1, 10]
lams = [math.log(l) for l in lams]
y_train_err, y_test_err = [], []
for lam in lams: 
    # Finds training error
    w_train = ridge_regression(lam, x_train, y_train)
    y_pred_train = np.dot(x_train, w_train)
    err_train = MSE(y_pred_train, y_train)
    y_train_err.append(err_train)
    
    # Finds testing error
    w_test = ridge_regression(lam, x_test, y_test)
    y_pred_test = np.dot(x_test, w_test)
    err_test = MSE(y_pred_test, y_test)
    y_test_err.append(err_test)
    
# Compute 5-cross fold validation and add the optimal lambda to results
lam = cross_val(5, x_train, y_train, lams) 
plot_err(lams, y_train_err, y_test_err, lam[1])


