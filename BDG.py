import numpy as np


def bgd(x,y,theta,alpha,m,maxx):
    x_trans=x.transpose()
    for i in range(0,maxx):
        hy=np.dot(x,theta)
        loss=hy-y
        gradient=np.dot(x_trans,loss)/m
        theta=theta-alpha*gradient
    return theta