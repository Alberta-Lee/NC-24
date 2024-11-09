import numpy as np
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import time

def load_file(file_name):
    with np.load(file_name) as data:
        xs = data['xs']
        ys = data['ys']
    return xs, ys

def cumulative_error_counts(y, y_pred):
    cumulative_error_count = np.cumsum(y != y_pred).astype(int)
    return cumulative_error_count

def cumulative_error_rates(y, y_pred):
    cumulative_error_count = cumulative_error_counts(y, y_pred)
    return cumulative_error_count / np.arange(1, cumulative_error_count.size+1)

def compute_precision_and_recall(y, y_pred):
    precision = []
    recall = []
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    bias = 0.000001
    for i in range(len(y)):
        if y[i] == 1:
            if y[i] != y_pred[i]:
                FN += 1
            else:
                TP += 1
        elif y[i] == -1:
            if y[i] != y_pred[i]:
                FP += 1
            else:
                TN += 1
        
        prec = TP / (TP+FP+bias)
        reca = TP / (TP+FN+bias)
        
        precision.append(prec)
        recall.append(reca)
    return precision, recall

def F_score(precision, recall):
    F_score = []
    for i in range(len(precision)):
        F = (2*precision[i]*recall[i]) / (precision[i]+recall[i])
        F_score.append(F)
    return F_score

def init_parameter(file_name):
    xs, ys = load_file(file_name)
    task_count, sample_count, sample_dim = xs.shape
    I = np.identity(task_count)
    omega = (1 / task_count) * I
    u = np.zeros(sample_dim)
    v = np.zeros((sample_dim, task_count))
    w = np.zeros((sample_dim, task_count))
    z = np.zeros((sample_dim, task_count))
    return u, v, w, z, omega

def MultiADMM(xs, ys, rho, eta, lambda1, lambda2, lambda3, lambda4, K=5):
    task_count, sample_count, sample_dim = xs.shape

    u, v, w, z, omega = init_parameter('../data/synthetic.npz')

    y_pred = np.full_like(ys, 0)

    for t in range(sample_count):
        start = time.time()
        for k in range(task_count):
            if t >= sample_count:
                continue
            x_instance = xs[k, t, :]
            y_instance = ys[k, t]

            y_pred[k, t] = np.sign((w[:, k].T @ x_instance).item())
            if y_pred[k, t] == 0:
                y_pred[k, t] = 1.0

            part = y_instance * y_pred[k, t]
            if part < 1.0:
                loss = 1 - part
            else:
                loss = 0.

            if part < 1.0:
                gradient = -y_instance * x_instance
            else:
                gradient = np.zeros(sample_dim)
            gradient = gradient.reshape(w[:, k].shape)
            w[:, k] = (eta/(rho+eta)) * w[:, k] + (rho/(rho+eta)) * (u + v[:, k]) - (1/(rho+eta))*(gradient + z[:, k])
        
        sum = np.zeros(sample_dim)
        for k in range(task_count):
            sum += (lambda1 + lambda3) * (z[:, k] + rho * w[:, k])
        u = (1/((lambda1+lambda3)*(lambda2+rho*K)+lambda2*rho)) * sum

        inv = np.linalg.pinv(omega)
        tmp = v @ inv + v @ np.transpose(inv)
        for k in range(task_count):
            v[:, k] = (lambda2/(lambda2*(lambda1+lambda3+rho)+rho*K*(lambda1+lambda3))) * (rho * w[:, k] + z[:, k]) + (lambda4/2)*tmp[:, k]
            z[:, k] += rho*(w[:, k] - u - v[:, k])

        mat = np.transpose(v) @ v
        mul = fractional_matrix_power(mat, 1/2)
        trace = np.trace(mul)
        omega = mul / trace

    return w, y_pred

if __name__ == '__main__':

    xs, ys = load_file('../data/synthetic.npz')
    
    task_count, sample_count, sample_dim = xs.shape

    rho = 0.5
    eta = 240
    lambda1 = 0.1
    lambda2 = 20
    lambda3 = 0.1
    lambda4 = 0.1
    K = 5

    w, y_pred = MultiADMM(xs, ys, rho, eta, lambda1, lambda2, lambda3, lambda4, K=5)

    cumulative_error_count_by_task = np.zeros((task_count, sample_count))
    cumulative_error_rate_by_task = np.zeros((task_count, sample_count))
    precision_by_task = np.zeros((task_count, sample_count))
    recall_by_task = np.zeros((task_count, sample_count))
    F1_score_by_task = np.zeros((task_count, sample_count))

    for k in range(task_count):
        cumulative_error_count_by_task[k, :] = cumulative_error_counts(y_pred[k, :], ys[k, :])
        cumulative_error_rate_by_task[k, :] = cumulative_error_rates(y_pred[k, :], ys[k, :])
        precision_by_task[k, :], recall_by_task[k, :] = compute_precision_and_recall(y_pred[k, :], ys[k, :])
        F1_score_by_task[k, :] = F_score(precision_by_task[k, :], recall_by_task[k, :])
    
    cumulative_error_count_mean = np.mean(cumulative_error_count_by_task, axis=0)
    cumulative_error_rate_mean = np.mean(cumulative_error_rate_by_task, axis=0)
    F1_score_mean = np.mean(F1_score_by_task, axis=0)

    print(cumulative_error_rate_mean)