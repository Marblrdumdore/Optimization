import numpy as np


def nesterov(x_start, step, g, iterations = 100,discount=0.7):
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    for i in range(iterations):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        if abs(sum(grad)) < 1e-6:
            break
    return x