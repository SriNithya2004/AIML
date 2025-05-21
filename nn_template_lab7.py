import numpy as np

np.random.seed(42)

"""
Sigmoid activation applied at each node.
"""
def sigmoid(x):
    # cap the data to avoid overflow?
    # x[x>100] = 100
    # x[x<-100] = -100
    return 1/(1+np.exp(-x))

"""
Derivative of sigmoid activation applied at each node.
"""
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN:
    def __init__(self, input_dim, hidden_dim, activation_func = sigmoid, activation_derivative = sigmoid_derivative):
        """
        Parameters
        ----------
        input_dim : TYPE
            DESCRIPTION.
        hidden_dim : TYPE
            DESCRIPTION.
        activation_func : function, optional
            Any function that is to be used as activation function. The default is sigmoid.
        activation_derivative : function, optional
            The function to compute derivative of the activation function. The default is sigmoid_derivative.

        Returns
        -------
        None.

        """
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        # TODO: Initialize weights and biases for the hidden and output layers
        
        #numpy.random.normal(loc = 0.0, scale = 1.0, size = None)
        #loc   : [float or array_like]Mean of the distribution. 
        #scale : [float or array_like]StandardDerivation of the distribution. 
        #size  : [int or int tuples]. 
        #Output shape given as (m, n, k) then m*n*k samples are drawn. If size is None(by default), then a single value is returned
        self.n=input_dim
        self.d=hidden_dim
        self.weights1=np.random.normal(loc = 0.0, scale = 1.0, size = (self.d,self.n)).T  #2X4
        self.weights2=np.random.normal(loc = 0.0, scale = 1.0, size = (1,self.d)).T   #4X1
        self.bias1=np.random.normal(loc = 0.0, scale = 1.0, size = (1,self.d)).T   # 4X1
        self.bias2=np.random.normal(loc = 0.0, scale = 1.0, size = (1,1)) 
        
        # pass
        
    def forward(self, X):
        # Forward pass              X= 1000X2
        # TODO: Compute activations for all the nodes with the activation function applied 
        # for the hidden nodes, and the sigmoid function applied for the output node
        # TODO: Return: Output probabilities of shape (N, 1) where N is number of examples

        # weighted sum of inputs coming into each node (e.g., z1 = w1x + b1),

        # print(self.weights1.shape)
        # print(X.shape)
        # print(self.bias1.shape)
        z1 = self.weights1.T @ X.T + self.bias1         #4X1000
        # print(z1.shape)
        self.a1=sigmoid(z1)                             #4X1000
        # print(self.a1.shape)
        # print(self.weights2.shape)
        z2= self.weights2.T @ self.a1+self.bias2        #1X1000
        # print(z2.shape)
        a2=sigmoid(z2)                                  #1X1000
        # print(a2.shape)
        return a2.T                                     #1000X1
        # pass
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # TODO: Compute gradients for the output layer after computing derivative of sigmoid-based binary cross-entropy loss
        # TODO: When computing the derivative of the cross-entropy loss, don't forget to divide the gradients by N (number of examples)  
        # TODO: Next, compute gradients for the hidden layer
        # TODO: Update weights and biases for the output layer with learning_rate applied
        # TODO: Update weights and biases for the hidden layer with learning_rate applied
        N=X.shape[0]
        
        sigo=self.forward(X)  #1000X1
        y=y.reshape(-1,1)
        C = sigo-y  #(y^ -y)
        grad2=np.matmul(C.T,self.a1.T)/N   #a1 = 4x 1000  C=1000x1
        gradbias2=np.sum(C.T)/N


        # print(C.shape)
        # print(self.a1.shape)
        
        
        # Grad2 = np.mean(grad2, axis=1)
    
        # a1ones=np.ones((self.d,X.shape[0]))
        
        # print(gradbias2.shape)
        # print(self.weights1.shape)
        t1=C@self.weights2.T       #t1= 1000x1 x 1x4 == 1000x4
        t2=self.a1*(1-self.a1)     # 4x1000
        t3=t1*t2.T            #1000x4
        # print(t3.shape)
        # print(X.T.shape)
        grad1=(X.T)@t3/N    #1000x2
        # print(t1.shape)
        X1=np.ones((X.shape[0],1))
        gradbias1=(X1.T)@t3/X.shape[0] 
        # print(Grad1.shape)
        self.weights2=self.weights2-learning_rate*grad2.T
        self.weights1=self.weights1-learning_rate*grad1

        self.bias1 = self.bias1 - learning_rate*gradbias1.T
        self.bias2 = self.bias2 - learning_rate*(gradbias2.T)
        
        pass
        
    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            self.forward(X)
            # Backpropagation and gradient descent weight updates
            self.backward(X, y, learning_rate)
            # TODO: self.yhat should be an N times 1 vector containing the final
            # sigmoid output probabilities for all N training instances 
            self.yhat = self.forward(X)
            # print(self.yhat.shape)
            # TODO: Compute and print the loss (uncomment the line below)
            loss = np.mean(-y*np.log(self.yhat) - (1-y)*np.log(1-self.yhat))
            # TODO: Compute the training accuracy (uncomment the line below)
            accuracy = np.mean((self.yhat > 0.5).reshape(-1,) == y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            self.pred('pred_train.txt')
            
    def pred(self,file_name='pred.txt'):
        pred = self.yhat > 0.5
        with open(file_name,'w') as f:
            for i in range(len(pred)):
                f.write(str(self.yhat[i]) + ' ' + str(int(pred[i])) + '\n')

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    # Separate the data into X (features) and y (target) arrays
    X = data[:, :-1]
    y = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X.shape[1]
    hidden_dim = 4
    learning_rate = 0.05
    num_epochs = 100
    
    model = NN(input_dim, hidden_dim)
    model.train(X**2, y, learning_rate, num_epochs) #trained on concentric circle data 

    test_preds = model.forward(X_eval**2)
    model.pred('pred_eval.txt')

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Test accuracy: {test_accuracy:.4f}")
