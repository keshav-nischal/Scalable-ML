class Neuron():
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def get_activation(self, inputs):
        activation = 0
        for w, x in zip(self.weights, inputs):
            activation += w * x
        activation += self.bias
        return activation  


class Layer():
    def __init__(self, neurons):
        self.neurons = neurons
    
    def get_activation_vector(self, inputs):
        return [neuron.get_activation(inputs) for neuron in self.neurons]


# XOR input
x = [[0, 0], [0, 1], [1, 0], [1, 1]]

# hidden layer
neuron1 = Neuron([1, 1], -0.5)   # OR
neuron2 = Neuron([1, 1], -1.5)   # AND
hidden_layer = Layer([neuron1, neuron2])

# output layer
final_neuron = Neuron([1, -2], -0.5)
final_layer = Layer([final_neuron])

print(final_layer.get_activation_vector(hidden_layer.get_activation_vector([0, 0]))) #2
print(final_layer.get_activation_vector(hidden_layer.get_activation_vector([0, 1]))) #1
print(final_layer.get_activation_vector(hidden_layer.get_activation_vector([1, 0]))) #1
print(final_layer.get_activation_vector(hidden_layer.get_activation_vector([1, 1]))) #0
