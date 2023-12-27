import sys
sys.path.append('..')
import numpy as np
import time
import matplotlib.pyplot as plt
from common.np import *
from common.util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        """
        Initialize a Trainer instance.

        Args:
            model: The neural network model to train.
            optimizer: The optimizer used for updating model parameters during training.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        """
        Train the model.

        Args:
            x: Input data.
            t: Target labels.
            max_epoch (int): Maximum number of epochs for training. Defaults to 10.
            batch_size (int): Batch size for each iteration. Defaults to 32.
            max_grad (float, optional): Maximum gradient value for gradient clipping. Defaults to None.
            eval_interval (int, optional): Interval for printing evaluation results. If None, no evaluation is performed.
                Defaults to 20.
        """
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # Shuffle data
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]

                # Compute gradients and update parameters
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # Print evaluation results
                if eval_interval is not None and iters % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| Epoch %d | Iteration %d / %d | Time %d[s] | Loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        """
        Plot the training loss over iterations.

        Args:
            ylim (tuple, optional): Tuple specifying the y-axis limits. Defaults to None.
        """
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iteration (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


class RnnlmTrainer:
    """
    Trainer class for a Recurrent Nerual Network Language Model (RNNLM).

    This class handles the training process of the RNNLM including batching,
    loss calculation, and optimization.

    Attributes:
        model (Model): The nueral network model to be trained.
        optimizer (Optimizer): The optimizer to use for training.
        time_idx (int): Current time index for batch generation.
        ppl_list (list): List to store the perplexity after each evaluation.
        eval_interval (int): Interval of iterations to evaluate the model.
        current_epoch (int): Current epoch number of the training.
    """
    
    def __init__(self, model, optimizer):
        """
        Initializes the RnnlmTrainer.

        Args:
            model (Model): The RNN model to be trained.
            optimizer (Optimizer): The optimizer to use for training.
        """
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x: np.ndarray, t: np.ndarray, batch_size: int, time_size: int) -> tuple:
        """
        Generates a batch for training.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target data.
            batch_size (int): Size of the batch.
            time_size (int): Size of the time steps.

        Returns:
            tuple: 
                numpy.ndarray: Batch of input data.
                numpy.ndarray: Batch of target data.
        """
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # 배치에서 각 샘플을 읽기 시작하는 위치

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t
    
    def fit(self, xs: np.ndarray, ts: np.ndarray, max_epoch: int=10, batch_size: int=20,
            time_size: int=35, max_grad: float=None, eval_interval: int=20):
        """
        Trains the model.

        Args:
            xs (numpy.ndarray): Input data.
            ts (numpy.ndarray): Target data.
            max_epoch (int, optional): Maximum number of epochs. Defaults to 10.
            batch_size (int, optional): Size of the batch. Defaults to 20.
            time_size (int, optional): Size of the time steps. Defaults to 35.
            max_grad (float, optional): Maximum gradient for clipping. Defaults to None.
            eval_interval (int, optional): Interval of iterations to evaluate the model. Defaults to 20.
        """
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d | 반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iteration (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()

def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads