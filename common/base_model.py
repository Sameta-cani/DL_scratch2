import sys
sys.path.append('..')
import os
import pickle
from common.np import *
from common.util import to_gpu, to_cpu


class BaseModel:
    """
    Base class for all neural network models.

    This class provides basic functionalities for saving and laoding model parameters,
    as well as defining the forward and backward passes which should be implemented in derived classes.

    Attributes:
        params (list of np.ndarray): Parameters of the model.
        grads (list of np.ndarray): Gradients of the model's parameters.
    """

    def __init__(self):
        """
        Initializes the BaseModel.
        """
        self.params, self.grads = None, None

    def forward(self, *args):
        """
        Forward pass of the model.

        This method should be implemented in derived classes.

        Raises:
            NotImplementedError: If the method is not overridden in a derived class.
        """
        raise NotImplementedError
    
    def backward(self, *args):
        """
        Backward pass of the model.

        This model should be implemented in derived classes.

        Raise:
            NotImplementedError: If the method is not overridden in a derived class.
        """
        raise NotImplementedError
    
    def save_params(self, file_name: str=None):
        """
        Saves the model parameters to a file

        Args:
            file_name (str, optional): The name of the file to save the parameters.
                                       If None, the class name will be used.
                                       Defaults to None.
        """
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [param.astype(np.float16) for param in self.params]
        if GPU:
            params = [to_cpu(param) for param in params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)


    def load_params(self, file_name: str=None):
        """
        Loads the model parameters from a file.

        Args:
            file_name (str, optional): The name of the file from which to load the parameters.
                                        If None, the class name will be used.
                                        Defaults to None.
        
        Raises:
            IOError: If the specified file does not exist.
        """
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        file_path = os.path.join(os.getcwd(), file_name)
        if not os.path.exists(file_path):
            raise IOError('No file: ' + file_path)
        
        with open(file_path, 'rb') as f:
            params = pickle.load(f)

        params = [param.astype('f') for param in params]
        if GPU:
            params = [to_gpu(param) for param in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]
