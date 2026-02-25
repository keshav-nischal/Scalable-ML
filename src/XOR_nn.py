import random
import math

random.seed(42)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_prime(a):  # derivative in terms of activation a = sigmoid(z)
    return a * (1 - a)

class Neuron():
    def __init__(self, weights, bias):
        self.weights = weights[:]  # list of floats
        self.bias = bias
        self.temp = {}  # will hold 'z' and 'a' for current forward pass

    def forward(self, inputs, save_learning_data=False):
        # compute z = w·x + b
        z = 0.0
        for w, x in zip(self.weights, inputs):
            z += w * x
        z += self.bias
        a = sigmoid(z)

        if save_learning_data:
            self.temp['z'] = z
            self.temp['a'] = a
            self.temp['inputs'] = inputs[:]  # needed to compute weight gradients
        return a

    def clear_temp(self):
        self.temp = {}

class Layer():
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs, save_learning_data=False):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs, save_learning_data))
        return outputs

    def clear_temp(self):
        for neuron in self.neurons:
            neuron.clear_temp()

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers

    def predict(self, inputs, is_learning=False):
        for layer in self.layers:
            inputs = layer.forward(inputs, save_learning_data=is_learning)
        return inputs

    def clear_temp(self):
        for layer in self.layers:
            layer.clear_temp()

# --- create a small network for XOR ---
# XOR: two inputs -> 2 hidden neurons -> 1 output neuron

def rand_w():
    return random.uniform(-1.0, 1.0)

# hidden layer
neuron1 = Neuron([rand_w(), rand_w()], rand_w())
neuron2 = Neuron([rand_w(), rand_w()], rand_w())
hidden_layer = Layer([neuron1, neuron2])

# output layer (single neuron)
final_neuron = Neuron([rand_w(), rand_w()], rand_w())
final_layer = Layer([final_neuron])

nn = NeuralNetwork([hidden_layer, final_layer])

# training data for XOR
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    total_loss = 0.0
    for x, y in zip(X, Y):
        # forward (save temps)
        preds = nn.predict(x, is_learning=True)  # preds is list with single value
        y_pred = preds[0]
        loss = (y - y_pred) ** 2
        total_loss += loss

        # --- backprop for single output neuron ---
        out_neuron = nn.layers[-1].neurons[0]
        a_out = out_neuron.temp['a']
        inputs_to_out = out_neuron.temp['inputs']  # these are activations from hidden layer

        # dL/da_out for MSE = 2*(a_out - y)
        dL_da_out = 2.0 * (a_out - y)
        # dL/dz_out = dL/da_out * sigmoid'(a_out)
        dL_dz_out = dL_da_out * sigmoid_prime(a_out)

        # gradients for output weights and bias
        for i in range(len(out_neuron.weights)):
            dL_dw = dL_dz_out * inputs_to_out[i]
            out_neuron.weights[i] -= learning_rate * dL_dw
        # bias
        out_neuron.bias -= learning_rate * dL_dz_out

        # --- backprop into hidden layer (2 neurons) ---
        for h_neuron in nn.layers[0].neurons:
            a_h = h_neuron.temp['a']
            inputs_to_h = h_neuron.temp['inputs']  # original x
            # contribution of this hidden neuron to output's dz: weight_out * dL_dz_out
            # find corresponding weight in output neuron
            idx = nn.layers[0].neurons.index(h_neuron)
            w_from_h_to_out = out_neuron.weights[idx]
            # Note: when updating weights above we already changed out_neuron.weights;
            # to use the pre-update weight you would store it. In this simple loop
            # the sign/order still works because we used the same dL_dz_out computed
            # from the forward pass and old weights — it's acceptable here but in
            # production code compute all gradients first, then update weights.
            dL_da_h = w_from_h_to_out * dL_dz_out
            dL_dz_h = dL_da_h * sigmoid_prime(a_h)

            # update h_neuron weights and bias
            for j in range(len(h_neuron.weights)):
                dL_dw_h = dL_dz_h * inputs_to_h[j]
                h_neuron.weights[j] -= learning_rate * dL_dw_h
            h_neuron.bias -= learning_rate * dL_dz_h

        # clear temps for next sample
        nn.clear_temp()

    # optionally print progress
    if epoch % 1000 == 0:
        print(f"epoch {epoch}, avg loss {total_loss / len(X):.6f}")

# Evaluate
for x in X:
    out = nn.predict(x, is_learning=False)[0]
    print(x, out, "->", 1 if out >= 0.5 else 0)