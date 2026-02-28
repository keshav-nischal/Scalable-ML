import random
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Neuron():
    def __init__(self, weights, bias):
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        self.temp = {}

    def get_activation(self, inputs, save_learning_data=False):
        inputs = np.array(inputs, dtype=float)
        z = float(np.dot(self.weights, inputs) + self.bias)
        a = sigmoid(z)
        if save_learning_data:
            self.temp["pre_activation"] = z
            self.temp["activation"] = a
        return a

    def clear_temp(self):
        self.temp = {}

    def __str__(self):
        return f"{self.weights.tolist()}, {self.bias}, {self.temp}"

class Layer():
    def __init__(self, neurons):
        self.neurons = neurons

    def get_activation_vector(self, inputs, is_learning=False):
        return [neuron.get_activation(inputs, save_learning_data=is_learning) for neuron in self.neurons]

    def clear_temp(self):
        for neuron in self.neurons:
            neuron.clear_temp()

    def get_temp(self, key):
        return np.array([neuron.temp[key] for neuron in self.neurons], dtype=float)

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def predict(self, inputs, is_learning=False):
        for layer in self.layers:
            inputs = layer.get_activation_vector(inputs, is_learning)
        return np.array(inputs, dtype=float)

    def clear_temp(self):
        for layer in self.layers:
            layer.clear_temp()

# XOR data
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

random.seed(42)
np.random.seed(42)

# create network (keep ranges reasonable)
hidden = Layer([
    Neuron(np.random.uniform(-1, 1, size=2), np.random.uniform(-1, 1)),
    Neuron(np.random.uniform(-1, 1, size=2), np.random.uniform(-1, 1))
])
output = Layer([Neuron(np.random.uniform(-1, 1, size=2), np.random.uniform(-1, 1))])
nn = NeuralNetwork([hidden, output])

learning_rate = 0.5  # typical small value

target_epoch = 20000
batch_size = 100
curr_epoch = 0

while(curr_epoch<target_epoch):
    for i in range(batch_size):
        curr_epoch += 1
        for xi, yi in zip(x, y):
            # forward (store temps)
            out = nn.predict(xi, is_learning=True)  # out is array with single element
            # output layer values
            a_out = nn.layers[-1].get_temp("activation")[0]   # scalar
            a_hidden = nn.layers[-2].get_temp("activation")   # vector of hidden activations
            a_input = np.array(xi, dtype=float)               # input activations

            # dC/da (MSE)
            dc_da = 2.0 * (a_out - yi[0])   # scalar

            # da/dz for output (sigmoid derivative)
            da_dz_out = a_out * (1.0 - a_out)

            # dC/dz for output
            dC_dz_out = dc_da * da_dz_out  # scalar

            # gradients for output neuron's weights and bias
            grad_w_output = dC_dz_out * a_hidden     # vector (shape matches output weights)
            grad_b_output = dC_dz_out                # scalar

            # update output weights and bias (in-place)
            out_neuron = nn.layers[-1].neurons[0]
            out_neuron.weights -= learning_rate * grad_w_output
            out_neuron.bias    -= learning_rate * grad_b_output

            # ===== backprop into hidden layer =====
            # For each hidden neuron j:
            # dC/dz_j = w_output_j * dC/dz_out * a_j * (1 - a_j)
            w_out = out_neuron.weights  # vector of weights from hidden -> output
            dC_dz_hidden = (w_out * dC_dz_out) * (a_hidden * (1.0 - a_hidden))  # vector

            # gradients for hidden weights and biases
            # dC/dw_hidden_j = dC/dz_j * a_input  (a_input is vector)
            for j, hidden_neuron in enumerate(nn.layers[-2].neurons):
                grad_w_hj = dC_dz_hidden[j] * a_input
                grad_b_hj = dC_dz_hidden[j]
                hidden_neuron.weights -= learning_rate * grad_w_hj
                hidden_neuron.bias    -= learning_rate * grad_b_hj

            # clear temps for next sample
            nn.clear_temp()

nn.clear_temp()
# Print updated params (optional)
print("trained neurons:")
print("hidden layer")
for n in nn.layers[0].neurons:
    print(n)
print("output layer")
print(nn.layers[1].neurons[0])

for xi in x:
    print(nn.predict(xi))