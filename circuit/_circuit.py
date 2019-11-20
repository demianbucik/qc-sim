import math
from abc import abstractmethod
from collections import deque
from functools import reduce

import numpy as np

_default_circuit = None


class Circuit:
    def __init__(self):
        self.layers = []
        self.inputs = []
        self.input_state = None
        #self.gates = []
        #self.qubits = []
        #self.constants = []
        # self._as_default()
        self._compiled = False
        self.measure = None

    def __repr__(self):
        return (
            f'Quantum circuit:\n'
            f'\tConstants: {self.constants}\n'
            f'\tQubits: {self.qubits}\n'
            f'\tCircuit: {self.gates[-1]}\n'
            f'{self.constants[0]}'
        )
    # """

    def _as_default(self):
        global _default_circuit
        _default_circuit = self
    # """
    """
    def _get_ordered(self):
        # Returns an ordered list of gates.
        ordered = []
        q = deque(self.gates)
        # Set for faster lookups
        evalued = {*self.constants, *self.qubits}

        while q:
            c = q.popleft()
            if all(map(lambda x: x in evalued, c.inp)):
                ordered.append(c)
                evalued.add(c)
            else:
                q.append(c)

        return ordered
    """

    def compile(self):
        input_state = 1
        for inp in self.inputs:
            input_state = np.kron(input_state, inp)
        self.input_state = input_state
        for layer in self.layers:
            layer.mat = layer.eval()
        self._compiled = True

    def add_inputs(self, *inputs):
        self.inputs = inputs

    def add_layer(self, *gates):
        self.layers.append(Layer(*gates))

    def add_measure(self):
        self.measure = Measure()

    def run(self, feed_dict={}):
        if not self._compiled:
            self.compile()

        val = self.input_state
        for layer in self.layers:
            val = layer.mat @ val

        if self.measure is not None:
            val = self.measure(val)

        return val


class Layer:
    def __init__(self, *gates):
        self.gates = gates
        self.mat = None

    def eval(self):
        # return reduce(np.kron, self.gates)
        val = 1
        for g in self.gates:
            val = np.kron(val, g.mat)
        return val


class constant:
    def __init__(self, value):
        self.__value = value
        # _default_circuit.constants.append(self)

    def __repr__(self):
        return f'c{self.value()}'

    def value(self):
        return self.__value


class qubit:
    # Qubit placeholder
    def __init__(self):
        self.output = None
        # _default_circuit.qubits.append(self)

    def __repr__(self):
        return f'q{self.output}'


class Gate:
    def __init__(self):
        self.output = None
        self.name = 'Meta'
        # _default_circuit.gates.append(self)

    def __repr__(self):
        return f'{self.name} {self.inp} => {self.output}'

    @abstractmethod
    def eval(self):
        pass


class OracleMat(Gate):
    def __init__(self, mat, name='Oracle'):
        # mat mush be a unitary matrix, eg U* @ U = I, for real matrices: U.T @ U = I
        super().__init__()
        self.name = name
        self.mat = mat

    # def eval(self):
    #    return self.mat @ self.inp[0].output


class Not(Gate):
    def __init__(self):
        super().__init__([inp])
        self.name = 'Not'
        self.mat = np.array([
            [0, 1],
            [1, 0]
        ])

    # def eval(self):
    #    return self.mat @ self.inp[0].output


class Hadamard(Gate):
    def __init__(self, n_bits=1):
        super().__init__()
        self.name = 'H^%d' % n_bits
        self.n_bits = n_bits
        self.mat = self._get_mat()

    def _get_mat(self):
        h1 = 1/math.sqrt(2) * np.array([
            [1, 1],
            [1, -1]
        ])
        h = 1
        for _ in range(self.n_bits):
            h = np.kron(h, h1)
        return h

    # def eval(self):
    #    return self.mat @ self.inp[0].output


class Measure(Gate):
    """ Input is a n-qubit vector of size 2^n
        Output is a sampled state """

    def __init__(self):
        super().__init__()
        self.name = "Measure"

    def __call__(self, state):
        # Must hold: np.sum(state**2) == 1"
        # Returns a string representation of n sampled bits, for example: '0101'
        # Samples according to input qubit vector
        n_states = state.shape[0]
        probs = state ** 2
        sample = np.random.choice(n_states, p=probs)

        n_bits = str(int(np.log2(n_states)))
        bit_str_template = '{0:0' + n_bits + 'b}'
        return bit_str_template.format(sample)
