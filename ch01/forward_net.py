import numpy as np

class Sigmoid:
    def __init__(self):
        """
        Initialize the Sigmoid activation function.
        This class doesn't have any learnable parameters.

        Args:
            params (list): An empty list, as Sigmoid has no learnable parameters.
        """
        self.params = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    

class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        """
        Initialize the Affine layer with weights and biases.

        Args:
            W (numpy.ndarray): Weight matrix.
            b (numpy.ndarray): Bias vector.
        """
        self.params = [W, b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the Affine layer.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the Affine layer.
        """
        W, b = self.params
        out = np.dot(x, W) + b
        return out
    
class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a two-layer neural network.

        Args:
            input_size (int): Input size.
            hidden_size (int): Hidden layer size.
            output_size (int): Output size.
        """
        self.layers = [
            Affine(np.random.randn(input_size, hidden_size), np.random.randn(hidden_size)),
            Sigmoid(),
            Affine(np.random.randn(hidden_size, output_size), np.random.randn(output_size))
        ]

        self.params = [param for layer in self.layers for param in layer.params]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward pass to make predictions.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Predicted output.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)