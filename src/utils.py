import numpy as np
import sys
from scipy.stats import wasserstein_distance

def predict_label_by_sign(w, x):
    val = np.sign(w.T @ x)
    if val == 0:
        predict = 1.0
    else:
        predict = val
    return predict, val

def cal_loss_and_gradient(x, y, w):
    predict, val = predict_label_by_sign(w, x)
    part = y * predict

    if part < 1.0:
        loss = 1 - part
        gradient = -y * x
    else:
        loss = 0 
        gradient = np.zeros(len(x))
        gradient = gradient.reshape(w.shape)
    return loss, gradient

def calculate_gradient(x, y, w, lamda = 0.0001):
    part = np.exp(-y * x.dot(w))
    gradient = (-y * x * part) / (1 + part) + lamda * w
    return gradient

def sigmoid(x):
    if x >= 0:
        return np.longfloat(1.0 / (1 + np.exp(-x)))
    else:
        return np.exp(x)/(1 + np.exp(x))

def predict_label_by_sigmoid(w, x):
    val = sigmoid(w.T @ x)
    if val >= 0.5:
        predict = 1.0
    else:
        predict = -1.0
    return predict, val