import os
import sys
from itertools import permutations as pm
import math
import numpy as np
import random
from line import Circuit, CNOT_Gate, H_Gate, X_Gate
from line import *
from logzero import logger
from copy import deepcopy
from rotation import UU_CNOT_UU, UU_NOTC_UU, UU_Cal
from qiskit.circuit.library import QFT
import termplotlib as tpl

class QFTMutator(object):
    def __init__(self):
        self.type = "QFT"
        self.with_amplitude_amplification = False

    def inverse_QFT(self, num_qs):
        qft = QFT(num_qubits = num_qs, approximation_degree = 0, do_swaps = True, inverse = True, insert_barriers = False, name = "qft")
        return qft

    def add_amplitude_amplification(self, circuit):
        for i in range(circuit.num_qubits):
            circuit.add_gate(i, H_Gate())
        for i in range(circuit.num_qubits):
            circuit.add_gate(i, X_Gate())
        circuit.add_gate(circuit.num_qubits-1, H_Gate())
        controlled_idx = list(range(0, circuit.num_qubits-1))
        circuit.add_mcgate(controlled_idx, circuit.num_qubits-1)
        circuit.add_gate(circuit.num_qubits-1, H_Gate())
        for i in range(circuit.num_qubits):
            circuit.add_gate(i, X_Gate())
        for i in range(circuit.num_qubits):
            circuit.add_gate(i, H_Gate())

    def generate_circuit(self, circuit):
        num_qubits = circuit.num_qubits
        qft = self.inverse_QFT(num_qubits)
        new_circuit = deepcopy(circuit)
        for i in range(num_qubits):
            theta, phi, lamb = UU_Cal()
            new_circuit.add_u_gate(i, (theta, phi, lamb))
        new_circuit.add_QFT(qft)
        if self.with_amplitude_amplification == True:
            self.add_amplitude_amplification(new_circuit)


        return new_circuit

class RandomMutator(object):
    def __init__(self):
        self.seed_dict = {
            "h_gate": H_Gate(),
            "z_gate": Z_Gate(),
            "p_gate": P_Gate(),
            "rx_gate": RX_Gate(),
            "ry_gate": RY_Gate(),
            "rz_gate": RZ_Gate(),
            "t_gate": T_Gate(),
            "td_gate": TDG_Gate(),
            "s_gate": S_Gate()
        }
        self.controled_point = Controled_Point()
        self.num_all_gates = len(self.seed_dict)
        self.num_gates = 10

    def generate_circuit(self, circuit):
        stop_add_gate = False
        true_false_choice_list = [True, False]
        num_qubits = circuit.num_qubits
        line_index = [x for x in range(num_qubits)]
        new_circuit = deepcopy(circuit)
        num_gate = 0


        while not stop_add_gate:
            line = random.choice(line_index)
            gate_idx = random.choice(list(self.seed_dict.keys()))
            gate = deepcopy(self.seed_dict[gate_idx])
            if gate.has_parameter == True:
                gate.lamb = random.random()
            add_entangle = random.choice(true_false_choice_list)
            if add_entangle == True and gate.has_entangle == True:
                controled_line = random.choice([x for x in line_index if x != line])
                new_circuit.add_gate(line, gate, has_entangle = True, entangle_line_idx = controled_line)
            else:
                controled_line = 0
                new_circuit.add_gate(line, gate)
            num_gate += 1
            if num_gate > int(circuit.num_qubits):
                stop_add_gate = True
        return new_circuit

class UCNOTMutator(object):
    def __init__(self):
        self.num_qubits = 0
        self.circuit = None

    def random_tf_machine(self):
        if random.randint(0,1) == 0:
            return False
        return True

    def uu_gate(self, idx, circuit):
        theta, fi, lamb = UU_Cal()
        circuit.add_u_gate(idx, (theta, fi, lamb))

    def uu_cnot_uu(self, idx_1, idx_2, circuit):
        u1 = (UU_Cal())
        u2 = (UU_Cal())
        u3 = (UU_Cal())
        u4 = (UU_Cal())
        circuit.add_u_gate(idx_1, u1)
        circuit.add_u_gate(idx_2, u2)
        circuit.add_gate(idx_2, CNOT_Gate(), True, idx_1)
        circuit.add_u_gate(idx_1, u3)
        circuit.add_u_gate(idx_2, u4)
    
    def uu_notc_uu(self, idx_1, idx_2, circuit):
        u1 = (UU_Cal())
        u2 = (UU_Cal())
        u3 = (UU_Cal())
        u4 = (UU_Cal())
        circuit.add_u_gate(idx_1, u1)
        circuit.add_u_gate(idx_2, u2)
        circuit.add_gate(idx_1, CNOT_Gate(), True, idx_2)
        circuit.add_u_gate(idx_1, u3)
        circuit.add_u_gate(idx_2, u4)

    def generate_u_gate_list(self, num_qubits):
        if num_qubits == 1:
            num_u_gates = 1
            num_ucnot_gates = 0

        elif num_qubits == 2:
            if self.random_tf_machine() == True:
                num_u_gates = 2
                num_ucnot_gates = 0
            else:
                num_u_gates = 0
                num_ucnot_gates = 2
        else:
            num_ucnot_gates = math.ceil(num_qubits / 2.5)
            if num_ucnot_gates % 2 != 0:
                num_ucnot_gates += 1
            num_u_gates = num_qubits - num_ucnot_gates
        idx_list = list(range(num_qubits))
        u_idx_list = random.sample(idx_list, num_u_gates)
        uucnot_idx_list = list(set(idx_list) - set(u_idx_list))
        return u_idx_list, uucnot_idx_list

    def generate_circuit(self, old_circuit):
        new_circuit = deepcopy(old_circuit)
        num_qubits = new_circuit.num_qubits
        u_idx_list, uucnot_idx_list = self.generate_u_gate_list(num_qubits)
        for u in u_idx_list:
            self.uu_gate(u, new_circuit)
        uucnot_idx_list_copy = deepcopy(uucnot_idx_list)
        while len(uucnot_idx_list_copy) > 0:
            idx_1 = uucnot_idx_list_copy.pop()
            idx_2 = uucnot_idx_list_copy.pop()
            choice = self.random_tf_machine()
            if choice == True:
                self.uu_cnot_uu(idx_1, idx_2, new_circuit)
            elif choice == False:
                self.uu_notc_uu(idx_1, idx_2, new_circuit)
        return new_circuit

class UCNOTMutator2(object):
    def __init__(self):
        self.num_qubits = 0
        self.circuit = None

    def random_tf_machine(self):
        if random.randint(0,1) == 0:
            return False
        return True

    def uu_gate(self, idx, circuit):
        theta, fi, lamb = UU_Cal()
        theta = theta / 100
        fi = fi / 80
        lamb = lamb / 50
        circuit.add_u_gate(idx, (theta, fi, lamb))

    def uu_cnot_uu(self, idx_1, idx_2, circuit):
        u1 = (UU_Cal())
        u2 = (UU_Cal())
        u3 = (UU_Cal())
        u4 = (UU_Cal())
        circuit.add_u_gate(idx_1, u1)
        circuit.add_u_gate(idx_2, u2)
        circuit.add_gate(idx_2, CNOT_Gate(), True, idx_1)
        circuit.add_u_gate(idx_1, u3)
        circuit.add_u_gate(idx_2, u4)
    
    def uu_notc_uu(self, idx_1, idx_2, circuit):
        u1 = (UU_Cal())
        u2 = (UU_Cal())
        u3 = (UU_Cal())
        u4 = (UU_Cal())
        circuit.add_u_gate(idx_1, u1)
        circuit.add_u_gate(idx_2, u2)
        circuit.add_gate(idx_1, CNOT_Gate(), True, idx_2)
        circuit.add_u_gate(idx_1, u3)
        circuit.add_u_gate(idx_2, u4)

    def generate_u_gate_list(self, num_qubits):
        if num_qubits == 1:
            num_u_gates = 1
            num_ucnot_gates = 0

        elif num_qubits == 2:
            if self.random_tf_machine() == True:
                num_u_gates = 2
                num_ucnot_gates = 0
            else:
                num_u_gates = 0
                num_ucnot_gates = 2
        else:
            num_ucnot_gates = math.ceil(num_qubits / 2.5)
            if num_ucnot_gates % 2 != 0:
                num_ucnot_gates += 1
            num_u_gates = num_qubits - num_ucnot_gates
        idx_list = list(range(num_qubits))
        u_idx_list = random.sample(idx_list, num_u_gates)
        uucnot_idx_list = list(set(idx_list) - set(u_idx_list))
        return u_idx_list, uucnot_idx_list

    def generate_circuit(self, old_circuit):
        new_circuit = deepcopy(old_circuit)
        num_qubits = new_circuit.num_qubits
        u_idx_list, uucnot_idx_list = self.generate_u_gate_list(num_qubits)
        for u in u_idx_list:
            self.uu_gate(u, new_circuit)
        uucnot_idx_list_copy = deepcopy(uucnot_idx_list)
        while len(uucnot_idx_list_copy) > 0:
            idx_1 = uucnot_idx_list_copy.pop()
            idx_2 = uucnot_idx_list_copy.pop()
            choice = self.random_tf_machine()
            if choice == True:
                self.uu_cnot_uu(idx_1, idx_2, new_circuit)
            elif choice == False:
                self.uu_notc_uu(idx_1, idx_2, new_circuit)
        return new_circuit
    
class AllHadmard(object):
    def __init__(self):
        self.num_qubits = 0
        self.circuit = None

    def generate_circuit(self, old_circuit):
        new_circuit = deepcopy(old_circuit)
        for i in range(new_circuit.num_qubits):
            new_circuit.add_gate(i, H_Gate())
        return new_circuit

if __name__ == "__main__":
    circuit = Circuit(4)
    mutator = QFTMutator()
    new_circuit = mutator.generate_circuit(circuit)
    print(new_circuit.code)
    print(new_circuit.results)
