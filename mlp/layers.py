import numpy as np
import matplotlib.pyplot as plt

class LinearLayer():
    """
    Function y = A*x + B
    """
    def __init__(self, input_size, output_size) -> None:
        self.input_size = input_size
        self.output_size = output_size

        self.X = np.zeros([self.output_size, 1])
        self.A = np.random.rand(self.output_size, self.input_size) - np.array(0.5)
        self.B = np.random.rand(self.output_size, 1)

        self.A_grad = np.zeros([self.output_size, self.input_size]) - np.array(0.5)
        self.B_grad = np.zeros([self.output_size, 1])
        self.X_grad = np.zeros([self.output_size, 1])
        self.grad_num = 0

    def forward(self, X):
        self.X = X
        return self.A@X + self.B
    
    def reset_grad(self):        
        self.A_grad = np.zeros([self.output_size, self.input_size])
        self.B_grad = np.zeros([self.output_size, 1])
        self.grad_num = 0
    
    def backward(self, DL_DY):
        #DL/DA = DL/DY * DY/DA
        if DL_DY.ndim == 1:
            DL_DY = np.column_stack(DL_DY)
        self.A_grad += (self.X @ DL_DY.T).T
        self.B_grad += DL_DY
        self.X_grad = (DL_DY.T @ self.A).T
        self.grad_num += 1
        return self.X_grad

    def update(self, learning_rate): 
        self.A -= self.A_grad * learning_rate
        self.B -= self.B_grad * learning_rate

class ReLU():
    """
    Function yi = xi, if xi > 0
             yi = 0, if xi <= 0
    """
    def __init__(self) -> None:
        pass

    def forward(self, X):
        self.X = X
        self.X_grad = np.zeros(self.X.shape)
        Y = X
        for i in range(x.shape[1]):
            if X[0][i] <= 0:
                Y[0][i] = 0
        return Y

    def reset_grad(self):
        self.X_grad = np.zeros(self.X.shape)

    def backward(self, DL_DY):
        self.X_grad = np.heaviside(self.X, 1) * DL_DY
        return self.X_grad
    
    def update(self, learning_rate): 
        pass

class ReLU_mod():
    """
    Function yi = xi, if xi > 0
             yi = 0, if xi <= 0
    """
    def __init__(self) -> None:
        pass

    def forward(self, X):
        self.X = X
        self.X_grad = np.zeros(self.X.shape)
        Y = X
        for i in range(x.shape[1]):
            if X[0][i] <= 0:
                Y[0][i] = Y[0][i]*0.1
        return Y

    def reset_grad(self):
        self.X_grad = np.zeros(self.X.shape)

    def backward(self, DL_DY):
        self.X_grad = (np.heaviside(self.X, 1)*1.1 - 0.1) * DL_DY 
        return self.X_grad
    
    def update(self, learning_rate): 
        pass

class Tanh():
    """
    Function yi = xi, if xi > 0
             yi = 0, if xi <= 0
    """
    def __init__(self) -> None:
        pass

    def forward(self, X):
        self.X = X
        self.X_grad = np.zeros(self.X.shape)
        Y = np.tanh(X)
        return Y

    def reset_grad(self):
        self.X_grad = np.zeros(self.X.shape)

    def backward(self, DL_DY):
        self.X_grad = (1-np.tanh(self.X)**2) * DL_DY 
        return self.X_grad
    
    def update(self, learning_rate): 
        pass

class Quadratic_Loss():
    """
    Loss
    """
    def __init__(self) -> None:
        pass

    def forward(self, Y_hat, Y):
        self.loss = 0
        for i in range(Y.shape[0]):
            self.loss += 0.5 * (Y_hat[i] - Y[i]) ** 2
        return self.loss

    def backward(self, Y_hat, Y):
        return (Y_hat - Y)

def get_data():
    X = np.linspace(-1, 1, 20)
    X.shape =  X.shape[0] , 1
    Y = X**3 + X**2
    return X, Y


if __name__ == "__main__":
    np.random.seed(42)
    X, Y = get_data()
    Y_hat = np.zeros(Y.shape)       
    # model = [LinearLayer(1,1)]
    # model = [LinearLayer(1,2),LinearLayer(2,1)]
    model = [LinearLayer(1,100),Tanh(),LinearLayer(100,3),Tanh(),LinearLayer(3,1)]
    loss = Quadratic_Loss()
    learning_rate = 0.0001
    errors = []
    for index in range(30000):
        error = 0
        for i in range(X.shape[0]):    
            x = np.array(X[i], ndmin=2)
            # x.shape = (1,1)
            y = np.array(Y[i], ndmin=2)
            # y.shape = (1,1)

            x_i = x
            for j in range(len(model)):
                x_i = model[j].forward(x_i)
            Y_hat[i] = x_i
            error += loss.forward(Y_hat[i], y)

            #TODO FIX ME
            DL_Yj = loss.backward(Y_hat[i], y)
            for j in reversed(range(len(model))):
                DL_Yj = model[j].backward(DL_Yj)
    
        for j in range(len(model)):
            model[j].update(learning_rate)        
            model[j].reset_grad()
        # print( model[4].A)
        print("Loss: ", error)
        errors.append(error)

        if np.isnan(error):
            break
        # input()

    plt.plot(errors)
    plt.show()

    plt.clf()
    plt.plot(X, Y, 'bo') 
    plt.plot(X, Y_hat, 'yo')

    plt.show()