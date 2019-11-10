import numpy as np
from circuit import Circuit, constant, Not

if __name__ == "__main__":
    qc = Circuit()
    one = constant(np.array([0, 1]))
    not1 = Not(one)

    res = qc.run(not1)
    print(res)
    print(qc)
