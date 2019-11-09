from collections import deque
from abc import abstractmethod
import numpy as np


class Circuit:
    def __init__(self):
        self.computations = []
        self._as_default()

    def _as_default(self):
        global _default_circuit
        _default_circuit = self


class Session:
    def __init__(self):
        self.circuit = _default_circuit

    def get_ordered(self):
        """Returns an ordered list of computations."""
        ordered = []
        q = deque()
        for c in self.circuit.computations:
            if isinstance(c, Gate):
                q.append(c)
            else:
                ordered.append(c)
        evalued = set(ordered)  # For faster lookups

        while q:
            c = q.popleft()
            if all(map(lambda x: x in evalued, c.inputs)):
                ordered.append(c)
                evalued.add(c)
            else:
                q.append(c)

        return ordered

    def run(self, computation, feed_dict={}):
        computations = self.get_ordered()

        for c in computations:
            if isinstance(c, constant):
                c.output = c.value()
            elif isinstance(c, qubit):
                c.output = feed_dict[c]
            else:
                c.eval()

        return computation.output


class constant:
    def __init__(self, value):
        self.__value = value
        _default_circuit.computations.append(self)

    def value(self):
        return self.__value


class qubit:
    # Qubit placeholder
    def __init__(self):
        self.value = None
        _default_circuit.computations.append(self)


class Gate:
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None
        _default_circuit.computations.append(self)

    @abstractmethod
    def eval(self):
        pass


class Not(Gate):
    def __init__(self, inp):
        super().__init__([inp])
        self.mat = np.array([
            [0, 1],
            [1, 0]
        ])

    def eval(self):
        self.output = self.mat @ self.inputs[0].output


class Hadamard(Gate):
    def __init__(self, inp):
        pass

    def eval(self):
        pass


if __name__ == "__main__":
    circuit = Circuit()
    one = constant(np.array([0, 1]))
    not1 = Not(one)

    sess = Session()
    res = sess.run(not1)
    print(res)
