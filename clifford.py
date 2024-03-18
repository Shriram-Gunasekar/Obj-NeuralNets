import numpy as np

class CliffordNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
        self.weights = [self.random_clifford_element(prev_size, curr_size) for prev_size, curr_size in zip([input_size] + hidden_sizes, hidden_sizes)]
        self.biases = [self.random_clifford_element(size) for size in hidden_sizes]
        self.output_weights = self.random_clifford_element(hidden_sizes[-1], output_size)
        self.output_bias = self.random_clifford_element(output_size)

    def random_clifford_element(self, *shape):
        real_part = np.random.randn(*shape)
        vector_part = np.random.randn(*shape)
        bivector_part = np.random.randn(*shape)
        return real_part + vector_part + bivector_part

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        a = x
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W) + b
            a = self.sigmoid(z)
        output = np.dot(a, self.output_weights) + self.output_bias
        return self.sigmoid(output)

# Example usage
input_size = 3
hidden_sizes = [4, 5]
output_size = 2

model = CliffordNeuralNetwork(input_size, hidden_sizes, output_size)
input_data = np.array([0.1, 0.2, 0.3])
output = model.forward(input_data)
print(output)
