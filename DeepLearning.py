import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf #tensorflow is used for graphical representations for your computations
tf.disable_v2_behavior()

data =pd.read_csv('data_stocks.csv')
data = data.drop(['DATE'], axis=1)

#dimensions of dataset
n = data.shape[0]
p = data.shape[1]

#make a numpy arrat
data = data.values

#training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

#scale data
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

#build x and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

print(X_train.shape)

#number of features in training data
n_stocks = X_train.shape[1]
print(n_stocks)

#define X and Y as placeholders to store inputs and target data
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks]) #it contains the network output which is equal to stock prices at time t = t
#shape = 2 dimensional matrix, none means right now we do not know the number of observations that will flow through the neural network
Y = tf.placeholder(dtype=tf.float32, shape=[None]) #it contains the network output which is equal to stock prices at time t = t+1
#shape = 1 dimensional vector
#actual observed targets are in Y

#initilizers, which can be updated earlier during training
sigma = 1
weight_initializer= tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform',seed=None)
bias_initilizer = tf.zeros_initializer()
print(weight_initializer)
print(bias_initilizer)

#model architecture parameters
n_stocks = 500
n_neurons_1 = 1024 #the model contains 4 hidden layers and the subsequent layers are always half the size of the previous layer
n_neurons_2 = 512 # number of neurons
n_neurons_3 = 256 #half the size because the layer compresses the information it received from the previous layer
n_neurons_4 = 128
n_target = 1 #output assigned a single neuron

#Layer 1: variables for weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initilizer([n_neurons_1]))

#Layer 2: variables for weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2])) #the second dimensionof the previouslayer is the first dimension of the current layer
bias_hidden_2 = tf.Variable(bias_initilizer([n_neurons_2])) #the current layer

#Layer 3: variables for weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initilizer([n_neurons_3]))

#Layer 4: variables for weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initilizer([n_neurons_4]))

#output Layer : variables for weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initilizer([n_target]))

#high dimensional data can be dealt with neural networks through activation function
#hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1)) #relu = rectified linear unit (activation function)
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

#Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

#cost function (calculates the error between the actual and the predicted output
mse = tf.reduce_mean(tf.squared_difference(out, Y)) #mse = mean squared error

#Optimizer (update weight and bias) (it invokes the gradient which tells the direction in which the weight and bias have to be changed
opt = tf.train.AdamOptimizer().minimize(mse)

#Make session
net = tf.Session()

#run initializer
net.run(tf.global_variables_initializer())

#Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

#Number of epochs and batch size
epoch = 10
batch_size = 256

for e in range(epoch):
    #Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    #Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        #Run optimizer with batch
        net.run(opt, feed_dict={X:batch_x, Y:batch_y}) #using feed forward method

        #Show progress
        if np.mod(i,5) == 0:
            #Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'epoch ' + str(e) + ', Batch ' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)

#Print final MSE after training
mse_final = net.run(mse, feed_dict={X:X_test, Y:y_test})
print(mse_final)