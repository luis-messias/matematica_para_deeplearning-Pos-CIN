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
        self.A = np.random.rand(self.output_size, self.input_size)
        self.B = np.random.rand(self.output_size, 1)

        self.A_grad = np.zeros([self.output_size, self.input_size])
        self.B_grad = np.zeros([self.output_size, 1])
        self.X_grad = np.zeros([self.output_size, 1])
        self.grad_num = 0

    def forward(self, X):
        self.X = X
        return self.A@X + self.B
    
    def reset_grad(self):        
        self.A_grad = np.zeros([self.output_size, self.input_size])
        self.B_grad = np.zeros([self.output_size, 1])
    
    def backward(self, DL_DY):
        #DL/DA = DL/DY * DY/DA
        print(DL_DY)
        print(self.X)
        self.A_grad += DL_DY @ self.X
        self.B_grad += DL_DY
        self.X_grad = DL_DY @ self.A 
        self.grad_num += 1

    def update(self, learning_rate):
        self.A -= self.A_grad * learning_rate/self.grad_num
        self.B -= self.B_grad * learning_rate/self.grad_num

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
        self.X_grad = DL_DY  
        for i in range(self.X.shape[1]):
            if self.X[0][i] <= 0:
                self.X_grad[1][i] = 0
            else:
                self.X_grad[1][i] = 1


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

    def backward(self):
        return (Y_hat - Y)

def get_data():
    X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]])
    Y = X * 10 + 2
    return X, Y


if __name__ == "__main__":
    X, Y = get_data()
    Y_hat = np.zeros(Y.shape)       

    model = [LinearLayer(1,1)]
    loss = Quadratic_Loss()
    learning_rate = 0.2
    for index in range(10000):
        error = 0
        for i in range(X.shape[0]):    
            x = X[i]
            x.shape = (1,1)
            y = Y[i]
            y.shape = (1,1)

            x_i = x
            for j in range(len(model)):
                x_i = model[j].forward(x_i)
            Y_hat[i] = x_i
            error += loss.forward(Y_hat[i], y)

        for i in range(X.shape[0]):    
            DL_DY = loss.backward()
            DL_Yi = DL_DY[i]
            for i in range(len(model), 0, -1):
                DL_Yi = model[j].backward(DL_Yi)
        model[0].update(learning_rate)        
        print("Loss: ", error)
        # print(model[0].A_grad, model[0].B_grad)
        model[0].reset_grad()

        
    plt.plot(X, Y) 
    plt.plot(X, Y_hat, 'bo')

    plt.show()