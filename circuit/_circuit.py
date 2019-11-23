import numpy as np


class Circuit:
    def __init__(self):
        self.layers = []
        self.inputs = []
        self.input_state = None
        self._compiled = False
        self.measure = None
        self.mat = None
        self.state = None
        self.sample = None

    def __repr__(self):
        layers_str = '\n\t\t'.join(map(str, self.layers))
        return (
            f'Quantum circuit:\n'
            f'\tInputs: {self.inputs}\n'
            f'\tLayers: {layers_str}\n'
            f'\t{self.measure}'
        )

    def compile(self):
        for layer in self.layers:
            layer.mat = layer.eval()

        self.mat = self.layers[0].mat
        for layer in self.layers[1:]:
            self.mat = layer.mat @ self.mat

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

        for k in feed_dict.keys():
            k.vec = feed_dict[k]

        self.input_state = 1
        for inp in self.inputs:
            self.input_state = np.kron(self.input_state, inp.vec)

        val = self.input_state
        for layer in self.layers:
            val = layer.mat @ val

        self.state = val
        if self.measure is not None:
            val = self.measure(val)
            self.sample = val

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

    def __repr__(self):
        return f'{self.gates}'


class Qubit:
    # Can be used as a constant qubit, defined when building the circuit
    # Or a qubit placeholder, passed in feed_dict at runtime
    def __init__(self, vec=None):
        self.vec = vec

    def __repr__(self):
        return f'q{self.vec}'


class Measure():
    # Input is a quantum state vector of size 2^n
    # Output is a sampled state of n bits, represented with a string, eg. '0101'
    def __init__(self):
        self.name = "Measure"

    def __call__(self, state):
        # Must hold: np.sum(state**2) == 1"
        # Returns a string representation of n sampled bits, for example: '0101'
        # Samples according to probabilities specified with the input qubit (state)
        n_states = state.shape[0]
        probs = state ** 2
        sample = np.random.choice(n_states, p=probs)

        n_bits = str(int(np.log2(n_states)))
        bit_str_template = '{0:0' + n_bits + 'b}'
        return bit_str_template.format(sample)

    def __repr__(self):
        return f'{self.name}'
