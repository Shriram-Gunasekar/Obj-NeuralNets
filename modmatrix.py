import numpy as np

class ModularMatrixNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, modulus):
        self.modulus = modulus
        self.hidden_sizes = hidden_sizes
        self.weights = [self.random_modular_matrix(prev_size, curr_size) for prev_size, curr_size in zip([input_size] + hidden_sizes, hidden_sizes)]
        self.biases = [self.random_modular_matrix(1, size) for size in hidden_sizes]
        self.output_weights = self.random_modular_matrix(hidden_sizes[-1], output_size)
        self.output_bias = self.random_modular_matrix(1, output_size)

    def random_modular_matrix(self, rows, cols):
        return np.random.randint(0, self.modulus, size=(rows, cols))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        a = x
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W) + b
            a = self.sigmoid(z % self.modulus)  # Apply modular arithmetic
        output = np.dot(a, self.output_weights) + self.output_bias
        return self.sigmoid(output % self.modulus)  # Apply modular arithmetic

# Example usage
input_size = 3
hidden_sizes = [4, 5]
output_size = 2
modulus = 10  # Example modulus

model = ModularMatrixNeuralNetwork(input_size, hidden_sizes, output_size, modulus)
input_data = np.array([[0.1, 0.2, 0.3]])  # Example input data
output = model.forward(input_data)
print(output)
