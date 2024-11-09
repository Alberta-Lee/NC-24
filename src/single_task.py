import numpy as np
import matplotlib.pyplot as plt
import time


def load_file(file_name):
    with np.load(file_name) as data:
        xs = data['xs']
        ys = data['ys']
    
    x = np.zeros_like(xs)
    for i in range(xs.shape[0]):
        norm = np.linalg.norm(xs[i], axis=1, keepdims=True)
        x[i] = xs[i] / norm
    return x, ys

def cumulative_error_counts(y, y_pred):
    cumulative_error_count = np.cumsum(y != y_pred).astype(int)
    return cumulative_error_count

def cumulative_error_rates(y, y_pred):
    cumulative_error_count = cumulative_error_counts(y, y_pred)
    return cumulative_error_count / np.arange(1, cumulative_error_count.size+1)

def init_parameter(file_name):
    xs, ys = load_file(file_name)
    task_count, sample_count, sample_dim = xs.shape
    I = np.identity(task_count)
    omega = (1 / task_count) * I
    u = np.zeros((sample_dim, 1))
    v = np.zeros((sample_dim, 1))
    w = np.zeros((sample_dim, 1))
    z = np.zeros((sample_dim, 1))
    return u, v, w, z, omega

def SingleADMM(x, y, rho, eta, lambda1, lambda2, lambda3, lambda4, K=1):

    u, v, w, z, omega = init_parameter('../data/synthetic.npz')
    
    sample_count, sample_dim = x.shape
    y_pred = np.zeros_like(y)

    for t in range(sample_count):

        if t >= sample_count:
            continue
        x_instance = x[t, :]
        y_instance = y[t, :]

        y_pred[t, :] = np.sign((w.T @ x_instance).item())

        part = y_instance * y_pred[t, :]
        if part < 1.0:
            loss = 1 - part
        else:
            loss = 0.
    
        if y_pred[t, :] == 0:
            y_pred[t, :] = 1.0

        if part < 1.0:
            gradient = -y_instance * x_instance
        else:
            gradient = np.zeros(sample_dim)
        gradient = gradient.reshape(w.shape)
        w = (eta/(rho+eta)) * w + (rho/(rho+eta)) * (u + v) - (1/(rho+eta)) * (gradient + z)
        
        summ = np.zeros((sample_dim, 1))
        summ += (lambda1 + lambda3) * (z + rho*w)
        u = (1/((lambda1+lambda3)*(lambda2+rho*K)+lambda2*rho)) * summ

        v = (lambda2/(lambda2*(lambda1+lambda3+rho)+rho*K*(lambda1+lambda3))) * (z+rho*w) + lambda4 * v

        z += rho*(w - u - v)

    return y_pred

if __name__ == '__main__':
    xs, ys = load_file('../data/synthetic.npz')

    task_id = 0

    task_count, sample_count, sample_dim = xs.shape
    x = xs[task_id, : , :]
    y = ys[task_id, :].reshape((10000, -1))

    rho = 0.5
    eta = 20
    lambda1 = 0.5
    lambda2 = 20
    lambda3 = 0.5
    lambda4 = 0.5
    K = 1

    y_pred = SingleADMM(x, y, rho, eta, lambda1, lambda2, lambda3, lambda4, K=1)
    cum_error_rate = cumulative_error_rates(y, y_pred)
    print(cum_error_rate)