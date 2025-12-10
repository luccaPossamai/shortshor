
#################################
# Algoritmo de Shor             #
# implementado por L. Possamai  #
#################################
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


N = 15
a = 2
if math.gcd(a, N) != 1:
    raise ValueError("Escolha 'a' coprimo com N.")
n = math.ceil(math.log2(N))            # número de qubits no registrador alvo

# registradores nomeados
qr_phase = QuantumRegister(2 * n, name='phase')   # registrador de controle
qr_target = QuantumRegister(n, name='target')  # registrador que guarda |x>
classical = ClassicalRegister(n, name='measures')  # registrador bits classicos
classical_two = ClassicalRegister(
    2 * n, name='measures_anc')  # registrador que guarda |x>
qc = QuantumCircuit(qr_phase, qr_target, classical)
# Flipa o ultimo qbit dos targets (|1> = |0000...01>)
qc.x(qr_target[0])
# Hadamards no registrador de fase
qc.h(qr_phase)


def Uk_unitary(k, a, N, n):
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    a_k = pow(a, k, N)
    for x in range(dim):
        if x < N:
            y = (a_k * x) % N
        else:
            y = x
        U[y, x] = 1
    return UnitaryGate(U, label=f"U_{k}")


# Aplicar controlled-U^(2^j) com controle sendo qr_phase[j]
for j in range(2 * n):
    U = Uk_unitary(2**j, a, N, n)   # U^{2^j}
    CU = U.control()                 # gera gate controlado (1 controle)
    # ordem de qubits: [controle] + lista dos n alvos
    qc.append(CU, [qr_phase[j]] + [qr_target[i] for i in range(n)])

qc.measure(qr_target, classical)
print(qc.draw())


# simular
sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=1000).result()

counts = result.get_counts()

plot_histogram(counts)
plt.show()
