import numpy as np  

class Layer:
    def forward(self, x):
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros(output_size)

    def forward(self, x):
        return np.dot(self.weights, x) + self.bias

    def __repr__(self):
        return f"DenseLayer({self.input_size}, {self.output_size})"

class ReLULayer(Layer):
    def forward(self, x):
        return np.maximum(0, x)

    def __repr__(self):
        return "Relu()"

class SigmoidLayer(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def __repr__(self):
        return "Sigmoid()"
