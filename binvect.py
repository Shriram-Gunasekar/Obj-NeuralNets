import numpy as np

class BinaryVectorNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
        self.weights = [self.random_binary_vector(prev_size, curr_size) for prev_size, curr_size in zip([input_size] + hidden_sizes, hidden_sizes)]
        self.biases = [self.random_binary_vector(1, size) for size in hidden_sizes]
        self.output_weights = self.random_binary_vector(hidden_sizes[-1], output_size)
        self.output_bias = self.random_binary_vector(1, output_size)

    def random_binary_vector(self, rows, cols):
        return np.random.randint(0, 2, size=(rows, cols))

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

model = BinaryVectorNeuralNetwork(input_size, hidden_sizes, output_size)
input_data = np.array([[0, 1, 0]])  # Example binary input data
output = model.forward(input_data)
print(output)
