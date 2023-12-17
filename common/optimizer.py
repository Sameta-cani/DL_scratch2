import sys
sys.path.append('..')
from common.np import *


class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Args:
        lr (float): Learning rate for controlling the step size during optimization.

    Methods:
        __init__(lr=0.01): Initializes an SGD optimizer with a given learning rate.
        update(params, grads): Updates the parameters using the SGD optimization rule.
    """

    def __init__(self, lr: float=0.01):
        """
        Initializes an SGD optimizer with a given learning rate.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        self.lr = lr

    def update(self, params: list, grads: list):
        """
        Updates the parameters using the SGD optimization rule.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        for param, grad in zip(params, grads):
            param -= self.lr * grad


class Momentum:
    """
    Momentum optimizer for parameter updates during optimization.

    Args:
        lr (float): Learning rate for controlling the step size during optimization.
        momentum (float): Momentum coefficient, influencing the weight of previous updates.
        v (list): List of momentum terms for each parameter.
    """

    def __init__(self, lr: float=0.01, momentum: float=0.9):
        """
        Initializes a Momentum optimizer with given learning rate and momentum.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum coefficient. Defaults to 0.9.
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, parmas: list, grads: list):
        """
        Updates the parameters using the Momentum optimization rule.

        Args:
            parmas (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        if self.v is None:
            self.v = [np.zeros_like(param) for param in parmas]

        for param, grad ,v in zip(parmas, grads, self.v):
            v *= self.momentum
            v -= self.lr * grad
            param += v


class Nesterov:
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    Args:
        lr (float): Learning rate for controlling the step size during optimization.
        momentum (float): Momentum coefficient, influencing the weight of previous updates.
        v (list): List of momentum terms for each parameter.

    Methods:
        __init__(lr=0.01, momentum=0.9): Initializes a Nesterov optimizer with given learning rate and momentum.
        update(params, grads): Updates the parameters using the Nesterov optimization rule.
    """

    def __init__(self, lr: float=0.01, momentum: float=0.9):
        """
        Initializes a Nesterov optimizer with given learning rate and momentum.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum coefficient. Defaults to 0.9.
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: list, grads: list):
        """
        Updates the parameters using the Nesterov optimization rule.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        if self.v is None:
            self.v = [np.zeros_like(param) for param in self.params]

        for param, grad, v in zip(params, grads, self.v):
            v *= self.momentum
            v -= self.lr * grad
            param += self.momentum * self.momentum * self.v - (1 - self.momentum) * self.lr * grad


class AdaGrad:
    """
    AdaGrad optimizer for parameter updates during optimization.

    Args:
        lr (float): Learning rate for controlling the step size during optimization.
        h (list): List of accumulated squared gradients for each paramter.

    Methods:
        __init__(lr=0.01): Initializes an AdaGrad optimizer with a given learning rate.
        update(params, grads): Updates the parameters using the AdaGrad optimization rule.
    """

    def __init__(self, lr: float=0.01):
        """
        Initializes an AdaGrad optimizer with a given learning rate.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
        """
        self.lr = lr
        self.h = None

    def update(self, params: list, grads: list):
        """
        Updates the parameters using the AdaGrad optimization rule.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradient corresponding to the parameters.
        """
        if self.h is None:
            self.h = [np.zeros_like(param) for param in params]

        for param, grad, h in zip(params, grads, self.h):
            h += grad * grad
            param -= self.lr * grad / (np.sqrt(h) + 1e-7)


class RMSprop:
    """
    RMSprop optimizer for parameter updates during optimization.

    Args:
        lr (float): Learning rate for controlling the step size during optimization.
        decay_rate (float): Decay rate for controlling the exponential decay of squared gradients.
        h (list): List of accumulated squared gradients for each parameter.

    Methods:
        __init__(lr=0.01, decay_rate=0.99): Initializes an RMSprop optimizer with given learning rate and decay rate.
        update(params, grads): Updates the parameters using the RMSprop optimization rule.
    """

    def __init__(self, lr: float=0.01, decay_rate: float=0.99):
        """
        Initializes an RMSprop optimizer with given learning rate and decay rate.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            decay_rate (float, optional): Decay rate. Defaults to 0.99.
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params: list, grads: list):
        """
        Updates the paramters using the RMSprop optimization rule.

        Args:
            params (list): List of paramters to be updated.
            grads (list): List of gradients corresponding to the parameter.
        """
        if self.h is None:
            self.h = [np.zeros_like(param) for param in params]

        for param, grad, h in zip(params, grads, self.h):
            h *= self.decay_rate
            h += (1 - self.decay_rate) * grad * grad
            param -= self.lr * grad / (np.sqrt(self.h) + 1e-7)


class Adam:
    """
    Adam optimizer for paramter updates during optimization.

    Args:
        lr (float): Learning rate for controlling the step size during optimization.
        beta1 (float): Exponential decay rate for the first moment estimate.
        beta2 (float): Exponential decay rate for the second moment estiamte.
        iter (int): Iteration counter.
        m (list): List of first moment estimates for each paramter.
        v (list): List off second moment estimates for each paramter.

    Methods:
        __init__(lr=0.001, beta1=0.9, beta2=0.999): Initializes an Adam optimizer with given hyperparameters.
        update(params, grads): Updates the paramters using the Adam optimization rule.
    """

    def __init__(self, lr: float=0.001, beta1: float=0.9, beta2: float=0.999):
        """
        Initializes an Adam optimizer with given hyperparameters.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            beta1 (float, optional): Exponential decay rate for the first moment estimate. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimate. Defaults to 0.999.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: list, grads: list):
        """
        Updates the parameters using the Adam optimization rule.

        Args:
            params (list): List of parameters to be updated.
            grads (list): List of gradients corresponding to the parameters.
        """
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for param, grad, m, v in zip(params, grads, self.m, self.v):
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad**2 - v)

            param -= lr_t * m / (np.sqrt(v) + 1e-7)