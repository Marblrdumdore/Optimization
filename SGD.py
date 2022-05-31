import numpy as np

def stochastic_gradient_descent(data, theta, alpha, epoch):
    X0, y0 = get_fea_lab(data)
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(len(X0))
    avg_cost = np.zeros(epoch)

    for k in range(epoch):
        new_data = data.sample(frac=1)
        X, y = get_fea_lab(new_data)

        for i in range(len(X)):
            error = X[i:i + 1] * theta.T - y[i]
            cost[i] = computeCost(new_data, theta, i)

            for j in range(parameters):
                temp[0, j] = theta[0, j] - alpha * error * X[i:i + 1, j]

            theta = temp
        avg_cost[k] = np.average(cost)

    return theta, avg_cost