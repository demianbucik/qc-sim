import numpy as np
from circuit import Circuit, constant, Id, Not, Hadamard, OracleMat

if __name__ == "__main__":
    """
    We want to find out whether a function f: {0, 1}^n -> {0, 1} is balanced or constant
    Lets take a simple function f: f(0)=1, f(1) = 1
    We are working with qubits, so a bit with value 0 is represented by a vector [1, 0]
        and value 1 is represented by [0, 1]
    Matrix representation of this function acting on a qubit takes the form
    Nf = [[0, 0],
          [1, 1]].
    It holds: Nf @ zero = one
          and Nf @ one = one
        where zero, one are 0, 1 qubit vectors, @ is matrix multiplication
    To use a matrix (function) in a quantum circuit, it has to be unitary (U* @ U = I).
    To make the operation reversible (unitary matrix), we use a control bit.
    The corresponding unitary matrix takes the form
    Uf = [[0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0]]
    """

    Nf = np.array([
        [0, 0],
        [1, 1]
    ])

    # f = lambda x, y: (x, Xor(y, Nf @ x))

    Uf = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # To find out whether an unknown function f with a corresponding unitary matrix Uf is constant or balanced,
    # we use the Deutsch-Jozsa algorithm

    qc = Circuit()
    # Start with 2 zero constants qubits
    x = constant(np.array([1, 0]))
    y = constant(np.array([1, 0]))

    qc.add_inputs(x, y)
    qc.add_layer(Id(), Not())
    qc.add_layer(Hadamard(n_bits=2))
    qc.add_layer(OracleMat(mat=Uf))
    qc.add_layer(Hadamard(), Id())
    qc.add_measure()

    res = qc.run()
    print(res)
    print("---------")
    print(f'state: {qc.state}')
    print(f'sample: {qc.sample}')
