# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-16T12:25:45.523911Z","iopub.execute_input":"2024-10-16T12:25:45.524408Z","iopub.status.idle":"2024-10-16T12:25:48.878076Z","shell.execute_reply.started":"2024-10-16T12:25:45.524374Z","shell.execute_reply":"2024-10-16T12:25:48.876993Z"}}
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape
print(m, n)
np.random.shuffle(data)# Shuffle the data before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.  # Normalize features (RGB colors between 0 and 1)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-16T12:25:48.880788Z","iopub.execute_input":"2024-10-16T12:25:48.881253Z","iopub.status.idle":"2024-10-16T12:25:48.895832Z","shell.execute_reply.started":"2024-10-16T12:25:48.881212Z","shell.execute_reply":"2024-10-16T12:25:48.894442Z"}}
# Function to initialize parameters of the neural network
def init_params():
    # Initialize weights and biases randomly
    # 28*28=784 params in input, 10 hand-written numbers (classes)
    W1 = np.random.rand(10, 784) - 0.5  # rand pick a float between 0 and 1
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def init_params_2():
    # Initialize weights and biases randomly
    # 28*28=784 params in input, 10 hand-written numbers (classes)
    W1 = np.random.normal(0, 4/(784+10),(10, 784))  # pick a float following normale law between -1 and 1
    b1 = np.zeros((10, 1))
    W2 = np.random.normal(0, 4/(10+10),(10, 10))
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

# ReLU activation function
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax activation function
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 #1st layer
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2 #2nd layer
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivative of ReLU activation function
def ReLU_deriv(Z):
    return Z > 0

# Convert class labels to one-hot encoded vectors
def one_hot(Y):
    #Y.max()+1=9+1=10 get the class 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    #np.arrange(9)=[0 1 2 3 4 5 6 7 8]
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    #Derivate the loss function
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters using gradient descent
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-16T12:25:48.897184Z","iopub.execute_input":"2024-10-16T12:25:48.897565Z","iopub.status.idle":"2024-10-16T12:27:53.256191Z","shell.execute_reply.started":"2024-10-16T12:25:48.897530Z","shell.execute_reply":"2024-10-16T12:27:53.255089Z"}}
# Get predictions
def get_predictions(A2):
    #run through A2 probabilistic array and fetch for each line the index of the highest value
    return np.argmax(A2, 0)

# Calculate accuracy
def get_accuracy(predictions, Y):
#     print(predictions, Y)
    #count the number of predictions that are correct and turn it in a relative number
    return np.sum(predictions == Y) / Y.size

# Make predictions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Gradient descent optimization
def gradient_descent(X_train, Y_train, X_dev, Y_dev, alpha, iterations, init_func):
    L_accuracy_train = []
    L_accuracy_dev = []
    L_iteration = []
    W1, b1, W2, b2 = init_func()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_train, Y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Calculate training accuracy
        predictions_train = get_predictions(A2)
        accuracy_train = get_accuracy(predictions_train, Y_train)
        
        # Calculate dev accuracy
        predictions_dev = make_predictions(X_dev, W1, b1, W2, b2)
        accuracy_dev = get_accuracy(predictions_dev, Y_dev)
        
        L_iteration.append(i)
        L_accuracy_train.append(accuracy_train)
        L_accuracy_dev.append(accuracy_dev)

    return W1, b1, W2, b2, L_iteration, L_accuracy_train, L_accuracy_dev

# Train the model and compute both training and dev accuracies
W1, b1, W2, b2, L_iteration1, L_accuracy1, L_accuracy_dev1 = gradient_descent(X_train, Y_train, X_dev, Y_dev, 0.10, 500, init_params)
W1, b1, W2, b2, L_iteration2, L_accuracy2, L_accuracy_dev2 = gradient_descent(X_train, Y_train, X_dev, Y_dev, 0.10, 500, init_params_2)

# Plot training and dev accuracies
plt.plot(L_iteration1, L_accuracy1, label="Train Accuracy (init 1)")
plt.plot(L_iteration1, L_accuracy_dev1, label="Dev Accuracy (init 1)", linestyle='--')
plt.plot(L_iteration2, L_accuracy2, label="Train Accuracy (init 2)")
plt.plot(L_iteration2, L_accuracy_dev2, label="Dev Accuracy (init 2)", linestyle='--')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.title('Training and Development Accuracy')
plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-16T12:27:53.257508Z","iopub.execute_input":"2024-10-16T12:27:53.257847Z","iopub.status.idle":"2024-10-16T12:27:53.913919Z","shell.execute_reply.started":"2024-10-16T12:27:53.257810Z","shell.execute_reply":"2024-10-16T12:27:53.912715Z"}}
# Test predictions on sample images
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
# Predict and visualize the first few numbers
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(5, W1, b1, W2, b2)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-16T12:27:53.916603Z","iopub.execute_input":"2024-10-16T12:27:53.917085Z","iopub.status.idle":"2024-10-16T12:27:53.934452Z","shell.execute_reply.started":"2024-10-16T12:27:53.917044Z","shell.execute_reply":"2024-10-16T12:27:53.933142Z"}}
# Make predictions on the development set and compute accuracy
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print("Accuracy on development set:", accuracy)