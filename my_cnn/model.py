from my_cnn.layer import Conv, AvgPool, Fc
from my_cnn.activation import TanH, Softmax
from my_cnn.utils import *

class LeNet5():

    def __init__(self):
        self.conv1 = Conv(nb_filters = 6, kernel = 5, nb_channels = 1)
        self.tanh1 = TanH()
        self.pool1 = AvgPool(kernel = 2, stride = 2)
        self.conv2 = Conv(nb_filters = 16, kernel = 5, nb_channels = 6)
        self.tanh2 = TanH()
        self.pool2 = AvgPool(kernel = 2, stride = 2)
        self.pool2_shape = None
        self.fc1 = Fc(row = 120, column = 5*5*16)
        self.tanh3 = TanH()
        self.fc2 = Fc(row = 84, column = 120)
        self.tanh4 = TanH()
        self.fc3 = Fc(row = 10 , column = 84)
        self.softmax = Softmax()

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]


    def forward(self, X):
        conv1 = self.conv1.forward(X)
        act1 = self.tanh1.forward(conv1)
        pool1 = self.pool1.forward(act1)

        conv2 = self.conv2.forward(pool1)
        act2 = self.tanh2.forward(conv2)
        pool2 = self.pool2.forward(act2)
        
        self.pool2_shape = pool2.shape
        pool2_flatten = pool2.reshape(self.pool2_shape[0], -1)
    
        fc1 = self.fc1.forward(pool2_flatten)
        act3 = self.tanh3.forward(fc1)
        
        fc2 = self.fc2.forward(act3)
        act4 = self.tanh4.forward(fc2)
        
        fc3 = self.fc3.forward(act4)
    
        y_pred = self.softmax.forward(fc3)

        return y_pred
        
    def backward(self, y_pred, y):
        deltaL = self.softmax.backward(y_pred, y)
        deltaL, dW5, db5, = self.fc3.backward(deltaL)
        deltaL = self.tanh4.backward(deltaL) 
        
        deltaL, dW4, db4 = self.fc2.backward(deltaL)
        deltaL = self.tanh3.backward(deltaL) 
        
        
        deltaL, dW3, db3 = self.fc1.backward(deltaL) 
        deltaL = deltaL.reshape(self.pool2_shape) 
        
        deltaL = self.pool2.backward(deltaL) 
        deltaL = self.tanh2.backward(deltaL)
        
        deltaL, dW2, db2 = self.conv2.backward(deltaL) 
        
        deltaL = self.pool1.backward(deltaL) 
        deltaL = self.tanh1.backward(deltaL)
        
        deltaL, dW1, db1 = self.conv1.backward(deltaL) 
        grads = { 
                'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2, 
                'dW3': dW3, 'db3': db3,
                'dW4': dW4, 'db4': db4,
                'dW5': dW5, 'db5': db5
        }
        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['W' + str(i+1)] = layer.W['val']
            params['b' + str(i+1)] = layer.b['val']

        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W['val'] = params['W'+ str(i+1)]
            layer.b['val'] = params['b' + str(i+1)]
