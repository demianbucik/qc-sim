import numpy as np
from circuit import Circuit, Qubit, Id, X, H, Oracle

if __name__ == "__main__":
    """
    We want to find out whether a function f: {0, 1}^n -> {0, 1} is balanced or constant
    Lets take a simple function f: f(0)=1, f(1) = 1
    We are working with qubits, so a bit with value 0 is represented with a vector [1, 0]
        and bit 1 is represented with [0, 1]
    Matrix representation of this function acting on a single qubit takes the form
    Nf = [[0, 0],
          [1, 1]].
    It holds: Nf @ zero = one
          and Nf @ one = one
        where zero, one are 0, 1 qubit vectors and @ is matrix multiplication
    To use a function (matrix) in a quantum circuit, it has to be unitary (U* @ U = I).
    To make the operation reversible (unitary matrix), we use a control bit and define a new function:
    f = lambda x, y: (x, Xor(y, Nf @ x))
    The corresponding unitary matrix takes the form
    Uf = [[0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0]]
    """
    Uf = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # To find out whether an unknown function f with a corresponding unitary matrix Uf is constant or balanced,
    # we use the Deutsch-Jozsa algorithm

    qc = Circuit()
    # x - input qubit placeholder
    x = Qubit()
    # y - control qubit, set to 0
    y = Qubit(np.array([1, 0]))

    qc.add_inputs(x, y)
    qc.add_layer(Id(), X())
    qc.add_layer(H(n=2))
    qc.add_layer(Oracle(mat=Uf))
    qc.add_layer(H(), Id())
    qc.add_measure()

    # When running the circuit, you can pass in qubit values in a feed_dict
    # You can run it multiple times with different values passed, but it will only compile once
    res = qc.run(feed_dict={
        x: np.array([1, 0])
    })

    # Here res == qs.state
    print(qc)
    # The circuit is represented by a matrix:
    print(f'\nMatrix representation of the circuit:\n{qc.mat}\n')
    print(f'State: {qc.state}')
    print(f'Sample: {qc.sample}')
