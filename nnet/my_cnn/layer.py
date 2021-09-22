import numpy as np

from my_cnn.utils import im2col, col2im


class Conv():
    
    def __init__(self, nb_filters, kernel, nb_channels, stride=1, padding=0):
        self._n = nb_filters
        self._k = kernel
        self._c = nb_channels
        self._s = stride
        self._p = padding

        self.W, self.b = self._xavier_init()

        self._cache = None

    def forward(
        self,
        X,
    ):
        m, _, n_H_prev, n_W_prev = X.shape

        n = self._n
        n_H = int((n_H_prev + 2 * self._p - self._k)/ self._s) + 1
        n_W = int((n_W_prev + 2 * self._p - self._k)/ self._s) + 1
        
        X_col = im2col(X, self._k, self._k, self._s, self._p)
        w_col = self.W['val'].reshape((self._n, -1))
        b_col = self.b['val'].reshape(-1, 1)
        out = w_col @ X_col + b_col

        out = np.array(np.hsplit(out, m)).reshape((m, n, n_H, n_W))
        self._cache = X, X_col, w_col
        return out

    def backward(
        self,
        dy,
    ):
        X, X_col, w_col = self._cache
        m, _, _, _ = X.shape
        self.b['grad'] = np.sum(dy, axis=(0,2,3))
        dy = dy.reshape(dy.shape[0] * dy.shape[1], dy.shape[2] * dy.shape[3])
        dy = np.array(np.vsplit(dy, m))
        dy = np.concatenate(dy, axis=-1)
        dX_col = w_col.T @ dy
        dw_col = dy @ X_col.T
        dX = col2im(dX_col, X.shape, self._k, self._k, self._s, self._p)
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self._c, self._k, self._k))
                
        return dX, self.W['grad'], self.b['grad']

    def _xavier_init(self):
        W = {'val': np.random.randn(self._n, self._c, self._k, self._k) * np.sqrt(1. / (self._k)),
                  'grad': np.zeros((self._n, self._c, self._k, self._k))}  
        b = {'val': np.random.randn(self._n) * np.sqrt(1. / self._n), 'grad': np.zeros((self._n))}
        return W, b

class AvgPool():
    
    def __init__(self, kernel, stride=1, padding=0):
        self._k = kernel
        self._s = stride
        self._p = padding
        self._cache = None

    def forward(
        self,
        X,
    ):
        self._cache = X

        m, n_prev, n_H_prev, n_W_prev = X.shape
        n = n_prev
        n_H = int((n_H_prev + 2 * self._p - self._k)/ self._s) + 1
        n_W = int((n_W_prev + 2 * self._p - self._k)/ self._s) + 1
        
        X_col = im2col(X, self._k, self._k, self._s, self._p)
        X_col = X_col.reshape(n, X_col.shape[0]//n, -1)
        A_pool = np.mean(X_col, axis=1)
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n, n_H, n_W)

        return A_pool

    def backward(
        self,
        dy,
    ):
        X = self._cache
        m, n_prev, n_H_prev, n_W_prev = X.shape
        n = n_prev

        dy_flatten = dy.reshape(n, -1) / (self._k * self._k)
        dX_col = np.repeat(dy_flatten, self._k*self._k, axis=0)
        dX = col2im(dX_col, X.shape, self._k, self._k, self._s, self._p)

        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_prev))
        dX = dX.reshape(m, n_prev, n_H_prev, n_W_prev)
        return dX

class Fc():

    def __init__(self, row, column):
        self.row = row
        self.col = column

        self.W, self.b = self._xavier_init()
        
        self._cache = None

    def forward(
        self,
        fc,
    ):
        self._cache = fc
        A_fc = np.dot(fc, self.W['val'].T) + self.b['val']
        return A_fc

    def backward(
        self,
        deltaL,
    ):
        fc = self._cache
        m = fc.shape[0]
        self.W['grad'] = (1/m) * np.dot(deltaL.T, fc)
        self.b['grad'] = (1/m) * np.sum(deltaL, axis = 0)

        new_deltaL = np.dot(deltaL, self.W['val']) 
        return new_deltaL, self.W['grad'], self.b['grad']
    
    def _xavier_init(
        self,
    ):
        W = {'val': np.random.randn(self.row, self.col) * np.sqrt(1./self.col), 'grad': 0}
        b = {'val': np.random.randn(1, self.row) * np.sqrt(1./self.row), 'grad': 0}
        return W, b 

class CrossEntropyLoss():

    def __init__(self):
        pass
    
    def get(self, y_pred, y):
        loss = -np.sum(y * np.log(y_pred))
        return loss
