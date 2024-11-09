import numpy as np
import random
import math


def normalize(x):
    x_norm = np.linalg.norm(x)
    if x_norm > 0:
        x = x / x_norm
    return x

def init_vars(task_count, mu, sigma):
    a = np.zeros((task_count, 5))
    a[0,1:5] = 1
    theta = np.zeros((task_count, 1))
    for i in range(1, task_count):
        a[i,:] = a[i-1,:] + np.random.normal(mu, sigma, size=(1,5))
        theta[i, 0] = theta[i-1, 0] + np.random.normal(mu, sigma*math.pi*0.25, 1)
    return a, theta

def gen_synthetic_fourier(a, theta, normalize, task_id):
    # Generate an array with two elements of uniformly distributed numbers in the interval (-3, 3)
    x = -3 + 6 * np.random.rand(1, 2)
    # Rotate a vector by theta radian in the counterclockwise direction
    sin_val = np.sin(theta[task_id, 0])
    cos_val = np.cos(theta[task_id, 0])
    xx = x.dot(np.array([[cos_val, sin_val], [-sin_val, cos_val]]))
    # Calculate y
    xxx = xx[0, 0] - a[task_id, 0]
    h = a[task_id, 1] * np.sin(xxx) + \
        a[task_id, 2] * np.sin(2 * xxx) + \
        a[task_id, 3] * np.cos(xxx) + \
        a[task_id, 4] * np.cos(2 * xxx)
    y = np.sign(xx[0, 1] - h)
    if y == 0: y = 1
    # Map x
    x = np.array([x[0, 0],
                  x[0, 1],
                  x[0, 0] * x[0, 1],
                  x[0, 0] ** 2,
                  x[0, 1] ** 2,
                  x[0, 0] ** 3,
                  x[0, 1] ** 3,
                  x[0, 0] * x[0, 1] ** 2,
                  x[0, 0] ** 2 * x[0, 1]])
    # Normalize x as required
    if normalize:
        norm = np.linalg.norm(x)
        if norm != 0:
            x = x / norm
    return x, y

def gen_data(option):
    if option == 'synthetic_fourier':
        task_count = 5
        mu = 0  # mean
        sigma = 0.1  # standard deviation
        a, theta = init_vars(task_count, mu, sigma)

        while True:
            task_id = random.randint(0, task_count - 1)
            xs, ys = gen_synthetic_fourier(a, theta, True, task_id)
            yield xs, ys

if __name__ == '__main__':
    z = gen_data('synthetic_fourier')   # generator
    for i in z:
        print(i)