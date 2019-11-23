import math

import numpy as np


class Gate:
    def __init__(self, name, n=None):
        self.name = f'{name}'
        if n is not None and n > 1:
            self.name += f'^{n}'
        self.n = n
        self.mat = None

    def __repr__(self):
        return f'{self.name}'


class Oracle(Gate):
    # Oracle gate, an unitary matrix has to be passed in when constructing it
    def __init__(self, mat, name='U'):
        # unitary matrix: U* @ U = I, for real matrices: U.T @ U = I
        super().__init__(name)
        self.mat = mat


class Id(Gate):
    # Identity gate
    def __init__(self, n=1):
        super().__init__('I', n)
        self.mat = np.identity(2**n)


class X(Gate):
    # Not gate
    def __init__(self, n=1):
        super().__init__('X', n)
        self.mat = self._get_mat()

    def _get_mat(self):
        size = 2**self.n
        x = np.zeros((size, size))
        for i in range(size):
            x[i, size-i-1] = 1
        return x


class H(Gate):
    # Hadamard gate
    def __init__(self, n=1):
        super().__init__('H', n)
        self.mat = self._get_mat()

    def _get_mat(self):
        h1 = 1/math.sqrt(2) * np.array([
            [1, 1],
            [1, -1]
        ])
        h = 1
        for _ in range(self.n):
            h = np.kron(h, h1)
        return h
