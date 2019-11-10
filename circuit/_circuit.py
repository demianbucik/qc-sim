from collections import deque
from abc import abstractmethod
import numpy as np

_default_circuit = None


class Circuit:
    def __init__(self):
        self.gates = []
        self.qubits = []
        self.constants = []
        self._as_default()
        self._compiled = False

    def __repr__(self):
        return (
            f'Quantum circuit:\n'
            f'\tConstants: {self.constants}\n'
            f'\tQubits: {self.qubits}\n'
            f'\tCircuit: {self.gates[-1]}\n'
            f'{self.constants[0]}'
        )

    def _as_default(self):
        global _default_circuit
        _default_circuit = self

    def _get_ordered(self):
        """Returns an ordered list of gates."""
        ordered = []
        q = deque(self.gates)
        # Set for faster lookups
        evalued = {*self.constants, *self.qubits}

        while q:
            c = q.popleft()
            if all(map(lambda x: x in evalued, c.inputs)):
                ordered.append(c)
                evalued.add(c)
            else:
                q.append(c)

        return ordered

    def compile(self):
        self.gates = self._get_ordered()
        self._compiled = True

    def run(self, computation=None, feed_dict={}):
        if not self._compiled:
            self.compile()

        for c in self.constants:
            c.output = c.value()
        for q in self.qubits:
            q.output = feed_dict[q]
        for g in self.gates:
            g.output = g.eval()

        if computation is None:
            return self.gates[-1].output
        return computation.output


class constant:
    def __init__(self, value):
        self.__value = value
        _default_circuit.constants.append(self)

    def __repr__(self):
        return f'c{self.value()}'

    def value(self):
        return self.__value


class qubit:
    # Qubit placeholder
    def __init__(self):
        self.output = None
        _default_circuit.qubits.append(self)

    def __repr__(self):
        return f'q{self.output}'


class Gate:
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None
        self.name = 'Meta'
        _default_circuit.gates.append(self)

    def __repr__(self):
        return f'{self.name} {self.inputs} => {self.output}'

    @abstractmethod
    def eval(self):
        pass


class Not(Gate):
    def __init__(self, inp):
        super().__init__([inp])
        self.name = 'Not'
        self.mat = np.array([
            [0, 1],
            [1, 0]
        ])

    def eval(self):
        return self.mat @ self.inputs[0].output


class Hadamard(Gate):
    def __init__(self, inp):
        pass

    def eval(self):
        pass
