import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        self.coef_ = None #非theta0以外的参数
        self.intercept_ = None #截距，即theta0
        self._theta = None #所有theta，私有变量

    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        xb = np.hstack([np.ones((len(y_train),1)),X_train])
        self._theta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y_train)
        self.intercept_ = self._theta[:1]
        self.coef_ = self._theta[1:]

        return self

    def fit_bgd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):#该点的函数值（theta是点，x和y是已知常数）
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):#该点的梯度。这里的优化是对其整体向量化。
            #res = np.empty(len(theta))
            #res[0] = np.sum(X_b.dot(theta) - y)
            #for i in range(1, len(theta)):
            #    res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            #res = X_b.dot(theta) - y

            return 2. / len(X_b) *X_b.T.dot(X_b.dot(theta) - y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                last_theta = theta
                theta = theta - eta * dJ(theta, X_b, y)
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

        return self

    def fit_sgd(self, X_train, y_train, n_iters=50, t0=5, t1=50):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1, \
            'n_iters must greater than 0'

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot((X_b_i.dot(theta) - y_i)) * 2.

        def sgd(X_b, y, initial_theta, n_iters=5, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            #theta = initial_theta
            #for cur_iter in range(n_iters):
            #    rand_i = np.random.randint(len(X_b))
            #    gradient = dJ_sgd(theta, X_b[rand_i], y[rand_i])
            #    theta = theta - learning_rate(cur_iter) * gradient
            #return theta

            #为了保证所有样本都能被用到同样的次数，将索引打乱，来模拟随机抽取样本，这样既能保证随机，又能保证每个样本都能被用到
            #n_iters的值是所有样本都被用到一次，这样一轮的轮数，而不是之前的循环次数。

            theta = initial_theta
            m = len(X_b)
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes,:]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])#randn是标准正态分布中随机取值
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]


    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        xb = np.hstack([np.ones((len(X_predict),1)),X_predict])
        return xb.dot(self._theta)

    def score(self, X_test, y_test):
        y_preditc = self.predict(X_test)
        return r2_score(y_test, y_preditc)

    def __repr__(self):
        return 'LinearRegression()'