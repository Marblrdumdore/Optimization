import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import random


#2.2  批梯度下降法
def bgd(x,y,theta,alpha,m,maxx):
    x_trans=x.transpose()
    for i in range(0,maxx):
        hy=np.dot(x,theta)      #计算出预测的因变量结果值，m*2与2*1得到m*1的因变量矩阵
        loss=hy-y               #计算偏差值
        gradient=np.dot(x_trans,loss)/m  #计算当前的梯度(即偏导值)，有m个样本计算的是平均值
        theta=theta-alpha*gradient  #更新当前的theta，继续沿着梯度向下走
    return theta


#2.3  随机梯度下降法
def stochastic_gradient_descent(data, theta, alpha, epoch):
    X0, y0 = get_fea_lab(data)  # 提取X和y矩阵
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(len(X0))
    avg_cost = np.zeros(epoch)

    for k in range(epoch):
        new_data = data.sample(frac=1)  # 打乱数据
        X, y = get_fea_lab(new_data)  # 提取新的X和y矩阵

        for i in range(len(X)):
            error = X[i:i + 1] * theta.T - y[i]
            cost[i] = computeCost(new_data, theta, i)

            for j in range(parameters):
                temp[0, j] = theta[0, j] - alpha * error * X[i:i + 1, j]

            theta = temp
        avg_cost[k] = np.average(cost)

    return theta, avg_cost

#2.4  小批量梯度下降法
def mb_gradient_descent(train_data, theta, alpha, mb_size):
    X, y = get_fea_lab(train_data)
    temp = np.matrix(np.zeros(theta.shape))  # temp用于存放theta参数的值
    parameters = int(theta.shape[1])  # parameter用于存放theta参数的个数
    m = len(X)  # m用于存放数据集中的样本个数
    cost = np.zeros(int(np.floor(m / mb_size)))  # cost用于存放代价函数
    st_posi = list(np.arange(0, m, mb_size))  # st_posi用于存放每次小批量迭代开始的位置
    new_st_posi = st_posi[:len(cost)]  # 去掉最后一次小批量迭代开始的位置
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



#4.2  Nesterov加速梯度法
def nesterov(x_start, step, g, iterations = 100,discount=0.7):
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    for i in range(iterations):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)  # 计算更新后的优化点的梯度
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        print('[Epoch {0} grad={1}, x={2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x







#4,9    AMSGrad

f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)

xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2
x_list = np.arange(xmin, xmax + xstep, xstep)
y_list = np.arange(ymin, ymax + ystep, ystep)
x, y = np.meshgrid(x_list, y_list)
z = f(x, y)

df = lambda x: np.array( [2*(1.5 - x[0] + x[0]*x[1])*(x[1]-1) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(x[1]**2-1)
                                        + 2*(2.625 - x[0] + x[0]*x[1]**3)*(x[1]**3-1),
                           2*(1.5 - x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0] + x[0]*x[1]**2)*(2*x[0]*x[1])
                                         + 2*(2.625 - x[0] + x[0]*x[1]**3)*(3*x[0]*x[1]**2)])


def plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)

    ax.quiver(path[:-1,0], path[:-1,1], path[1:,0]-path[:-1,0], path[1:,1]-path[:-1,1], scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*minima_, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    plt.show()





#4.1  Momentum梯度下降法
def gradient_descent_momentum(df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-6):
    history = [x]
    v = np.zeros_like(x)  # 动量
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        v = gamma * v + alpha * df(x)  # 更新动量
        x = x - v  # 更新变量（参数）

        history.append(x)
    return history
# x0=np.array([3., 4.])
# print(x0)
# path = gradient_descent_momentum(df,x0,0.000005,0.8,300000)
# path = np.asarray(path)
# plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)

#4.3  Adagrad
def gradient_descent_Adagrad(df,x,alpha=0.01,iterations = 100,epsilon = 1e-8):
    history=[x]
    gl = np.ones_like(x)
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        gl += grad**2
        x = x-alpha* grad/(np.sqrt(gl)+epsilon)
        history.append(x)
    return history
# x0=np.array([3., 4.])
# path = gradient_descent_Adagrad(df,x0,0.1,300000,1e-8)
# path = np.asarray(path)
# plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)



#4.4  Adadelta
def gradient_descent_Adadelta(df,x,alpha = 0.1,rho=0.9,iterations = 100,epsilon = 1e-8):
    history=[x]
    Eg = np.ones_like(x)
    Edelta = np.ones_like(x)
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        Eg = rho*Eg+(1-rho)*(grad**2)
        delta = np.sqrt((Edelta+epsilon)/(Eg+epsilon))*grad
        x = x- alpha*delta
        Edelta = rho*Edelta+(1-rho)*(delta**2)
        history.append(x)
    return history
# x0=np.array([3., 4.])
# path = gradient_descent_Adadelta(df,x0,1.0,0.9,300000,1e-8)
# path = np.asarray(path)
# plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)


#4.5  RMSprop
def gradient_descent_RMSprop(df, x, alpha=0.01, beta=0.9, iterations=100, epsilon=1e-8):
    history = [x]
    v = np.ones_like(x)
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        v = beta * v + (1 - beta) * grad ** 2
        x = x - alpha * grad / (np.sqrt(v) + epsilon)

        history.append(x)
    return history
    x0=np.array([3., 4.])
    path = gradient_descent_RMSprop(df,x0,0.000005,0.99999999999,900000,1e-8)
    path = np.asarray(path)
    plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)


#4.6    Adam
def gradient_descent_Adam(df, x, alpha=0.01, beta_1=0.9, beta_2=0.999, iterations=100, epsilon=1e-8):
    history = [x]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad ** 2

        t = t + 1
        if True:
            m_1 = m / (1 - np.power(beta_1, t + 1))
            v_1 = v / (1 - np.power(beta_2, t + 1))
        else:
            m_1 = m / (1 - np.power(beta_1, t)) + (1 - beta_1) * grad / (1 - np.power(beta_1, t))
            v_1 = v / (1 - np.power(beta_2, t))

        x = x - alpha * m_1 / (np.sqrt(v_1) + epsilon)
        history.append(x)
    return history

    x0=np.array([3., 4.])
    path = gradient_descent_Adam(df,x0,0.001,0.9,0.8,100000,1e-8)
    path = np.asarray(path)
    plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)


#4.7    AdaMax
def gradient_descent_AdaMax(df, x, alpha=0.9, beta=0.999, iterations=100, epsilon=1e-8):
    history = [x]
    s = 0
    r = 0
    lr = 1e-3
    alpha_i = 1
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        s = s * alpha + (1 - alpha) * grad
        r = max(r * beta,np.max(np.abs(grad)))
        alpha_i *= alpha
        x = -lr / (1 - alpha_i) * s / r
        history.append(x)
    return history
# x0=np.array([3., 4.])
# path = gradient_descent_AdaMax(df,x0)
# path = np.asarray(path)
# plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)

# class AdaMax(object):
#     def __init__(self, lr=1e-3, alpha=0.9, beta=0.999):
#         self.s = 0
#         self.r = 0
#         self.lr = lr
#         self.alpha = alpha
#         self.alpha_i = 1
#         self.beta = beta
#
#     def update(self, g: np.ndarray):
#         self.s = self.s * self.alpha + (1 - self.alpha) * g
#         self.r = np.maximum(self.r * self.beta, np.abs(g))
#         self.alpha_i *= self.alpha
#         lr = -self.lr / (1 - self.alpha_i)
#         return lr * self.s / self.r



#4,8    Nadam
def gradient_descent_Nadam(df, x, alpha=0.9, beta=0.999, iterations=100, epsilon=1e-8):
    history = [x]
    s = 0
    r = epsilon
    lr = 1e-3
    alpha_i = 1
    beta_i = 1
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("梯度足够小！")
            break
        grad = df(x)
        s = s * alpha + (1 - alpha) * grad
        r = r * beta + (1 - beta) * np.square(grad)
        alpha_i *= alpha
        beta_i *= beta_i
        x = lr * (1 - beta_i) ** 0.5 / (1 - alpha_i) * (s * alpha + (1 - alpha) * grad) / np.sqrt(r)
        history.append(x)
    return history
x0 = np.array([3., 4.])
path = gradient_descent_Nadam(df,x0)
path = np.asarray(path)
plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)
# class Nadam(object):
#     def __init__(self, lr=1e-3, alpha=0.9, beta=0.999, eps=1e-8):
#         self.s = 0
#         self.r = eps
#         self.lr = lr
#         self.alpha = alpha
#         self.beta = beta
#         self.alpha_i = 1
#         self.beta_i = 1
#
#     def update(self, g: np.ndarray):
#         self.s = self.s * self.alpha + (1 - self.alpha) * g
#         self.r = self.r * self.beta + (1 - self.beta) * np.square(g)
#         self.alpha_i *= self.alpha
#         self.beta_i *= self.beta_i
#         lr = -self.lr * (1 - self.beta_i) ** 0.5 / (1 - self.alpha_i)
#         return lr * (self.s * self.alpha + (1 - self.alpha) * g) / np.sqrt(self.r)