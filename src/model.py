import numpy as np
from layers import DenseLayer, ReLULayer, SigmoidLayer

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __repr__(self):
        return f"Model({self.layers})"

if __name__ == "__main__":
    model = Model([
        DenseLayer(2, 4),
        ReLULayer(),
        DenseLayer(4, 2),
        SigmoidLayer()
    ])

    x = np.array([1.0, 2.0])
    print(model.forward(x))
    print(model)
