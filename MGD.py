import numpy as np

def mb_gradient_descent(train_data, theta, alpha, mb_size):
    X, y = get_fea_lab(train_data)
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    m = len(X)
    cost = np.zeros(int(np.floor(m / mb_size)))
    st_posi = list(np.arange(0, m, mb_size))
    new_st_posi = st_posi[:len(cost)]
    k = 0

    for i in new_st_posi:

        cost[k] = computeCost(train_data, theta, i, mb_size)
        k = k + 1
        error = (X * theta.T) - y

        for j in range(parameters):
            t = np.multiply(error, X[:, j])
            term = t[i:i + mb_size]
            temp[0, j] = theta[0, j] - (alpha / mb_size) * (np.sum(term))

        theta = temp

    return theta, cost