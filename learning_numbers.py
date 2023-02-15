import numpy as np
from tensorflow import keras
from keras.datasets import mnist
import random 
from tqdm import tqdm

class ANN:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b = [np.random.randn(x, 1) for x in sizes[1:]]

    def feedforward(self, x):
        a = x
        for layer in range(0, self.num_layers-1):
            w = self.w[layer]
            b = self.b[layer]
            z = np.dot(w, a) + b
            a = sigmoid(z)
        # Binarize output
        a[a>0.5] = 1
        a[a<=0.5] = 0
        return a

    def backprop(self, x, y):
        change_b = [np.zeros(p.shape) for p in self.b]
        change_w = [np.zeros(p.shape) for p in self.w]
        # Feedforward
        activations = [x]
        weighted_inputs = []
        a = x
        for layer in range(0, self.num_layers-1):
            w = self.w[layer]
            b = self.b[layer]
            z = np.dot(w, a) + b
            weighted_inputs.append(z)
            a = sigmoid(z)
            activations.append(a)
        # Output error delta for the last layer
        delta = (a - y) * sigmoid_prime(z)
        change_b[-1] = delta
        change_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            delta = np.dot(self.w[-layer+1].transpose(),delta) * sigmoid_prime(weighted_inputs[-layer])
            change_b[-layer] = delta
            change_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return change_w, change_b

    def update(self, train_data, learning_rate):
        total_change_w = [np.zeros(myW.shape) for myW in self.w]
        total_change_b = [np.zeros(myB.shape) for myB in self.b]
        for x, y in train_data:
            change_w, change_b = self.backprop(x,y)
            total_change_w = [w+dw for w, dw in zip(change_w, total_change_w)]
            total_change_b = [b+db for b, db in zip(change_b, total_change_b)]
        # Update weights and biases in NN
        self.w = [ow - (learning_rate/len(train_data))*dw for ow, dw in zip(self.w, total_change_w)]
        self.b = [ob - (learning_rate/len(train_data))*db for ob, db in zip(self.b, total_change_b)]


""" Activation functions """
def sigmoid(z):
    a = 1.0/(1.0 + np.exp(-1*z))
    # threshold = 0.5
    # x = np.array([1 if v > threshold else 0 for v in a])
    # a = np.vstack(x)
    return a
    
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1.0-sigmoid(z))

def perceptron(z):
    threshold = 0
    x = np.array([1 if v > threshold else 0 for v in z])
    a = np.vstack(x)
    return a

num_classes = 10
myNetwork = ANN([784, 20, num_classes])
# input = np.array([1,0]).reshape(2,1)
# label = np.array([1,0]).reshape(2,1)
# train_data = [(input, label)]
# myNetwork.update(train_data, learning_rate=0.1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

training_dataset = []
for i in range(60000):
    x = x_train[i].reshape(28*28,1)
    y = y_train[i].reshape(num_classes,1)
    training_dataset.append((x,y))

batch_size = 100
batch = []
previous_correctness = 0
platue = 0
for epochs in (range(60)):
    for i in range(600):
        batch = training_dataset[i*batch_size:(i+1)*batch_size]
        myNetwork.update(batch, learning_rate=2)

    #  Test
    correct = 0
    for i in range(10000):
        x = x_test[i].reshape(28*28,1)
        y = y_test[i].reshape(num_classes,1)
        y_result = myNetwork.feedforward(x)
        if np.array_equal(y_result, y):
            correct += 1
    correct = correct/10000
    print("Correctness = ", correct, ", Delta Correct = ", correct - previous_correctness)

    if (correct-previous_correctness) < 0.001:
        platue += 1
    else: 
        platue = 0
    
    if platue == 10:
        print(f"Learning has platued at epochs number {epochs}.")
        print("------------- FINAL TEST ----------------")
        correct = 0
        for i in range(10000):
            x = x_test[i].reshape(28*28,1)
            y = y_test[i].reshape(num_classes,1)
            y_result = myNetwork.feedforward(x)
            if np.array_equal(y_result, y):
                correct += 1
        correct = correct/10000
        print("Accuracy %: ", correct*100)
        break 
    previous_correctness = correct
    random.shuffle(training_dataset)

