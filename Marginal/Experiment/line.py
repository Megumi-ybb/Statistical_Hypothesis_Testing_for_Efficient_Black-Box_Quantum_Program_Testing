import os
import sys
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from logzero import logger
from copy import deepcopy
import random

def gen_bin_dict(num_qubits):
    output_list = list()
    output_list.append("0" * num_qubits)
    for i in range(1, pow(2, num_qubits)):
        tmp = list()
        r = 0
        while (i!=0):
            r = i%2
            i = i//2
            tmp = [str(r)] + tmp
        output_list.append("".join(tmp).zfill(num_qubits))
    output_dict = dict()
    for k in output_list:
        output_dict[k] = 0.0
    return output_dict

class H_Gate(object):
    def __init__(self):
        self.type = H_Gate
        self.has_entangle = True
        self.has_parameter = False

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        if has_entangle == True:
            code.ch(entangle_line_idx, idx)
        elif has_entangle == False:
            code.h(idx)
    
    def strs(self):
        return "|H |"

class X_Gate(object):
    def __init__(self):
        self.type = X_Gate
        self.has_entangle = False
        self.has_parameter = False
    
    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        code.x(idx)
    
    def strs(self):
        return "|X |"

class T_Gate(object):
    def __init__(self):
        self.type = T_Gate
        self.has_entangle = False
        self.has_parameter = False

    def add_code(self, code, idx):
        code.t(idx)

    def strs(self):
        return "|T |"

class TDG_Gate(object):
    def __init__(self):
        self.type = TDG_Gate
        self.has_entangle = False
        self.has_parameter = False

    def add_code(self, code, idx):
        code.tdg(idx)

    def strs(self):
        return "|T+|"

class S_Gate(object):
    def __init__(self):
        self.type = S_Gate
        self.has_entangle = False
        self.has_parameter = False
    
    def add_code(self, code, idx):
        code.s(idx)

    def strs(self):
        return "|S |"

class Z_Gate(object):
    def __init__(self):
        self.type = Z_Gate
        self.has_entangle = True
        self.has_parameter = False

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        if has_entangle == True:
            code.cz(entangle_line_idx, idx)
        elif has_entangle == False:
            code.z(idx)

    def strs(self):
        return "|Z |"

class P_Gate(object):
    def __init__(self, lamb=0.0):
        self.type = P_Gate
        self.lamb = lamb
        self.has_entangle = True
        self.has_parameter = True

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        lamb = random.randint(1,10)
        if has_entangle == True:
            code.cp(lamb, entangle_line_idx, idx)
        elif has_entangle == False:
            code.p(lamb, idx)

    def strs(self):
        return "|P |"

class RX_Gate(object):
    def __init__(self, lamb=0.0):
        self.type = RX_Gate
        self.lamb = lamb
        self.has_entangle = True
        self.has_parameter = True

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        lamb = random.randint(1,10)
        if has_entangle == True:
            code.crx(lamb, entangle_line_idx, idx)
        elif has_entangle == False:
            code.rx(lamb, idx)
    
    def strs(self):
        return "|RX|"

class RY_Gate(object):
    def __init__(self, lamb=0.0):
        self.type = RY_Gate
        self.lamb = lamb
        self.has_entangle = True
        self.has_parameter = True

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        lamb = random.randint(1,10)
        if has_entangle == True:
            code.cry(lamb, entangle_line_idx, idx)
        elif has_entangle == False:
            code.ry(lamb, idx)
    
    def strs(self):
        return "|RY|"

class RZ_Gate(object):
    def __init__(self, lamb=0.0):
        self.type = RZ_Gate
        self.lamb = lamb
        self.has_entangle = True
        self.has_parameter = True

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        lamb = random.randint(1,10)
        if has_entangle == True:
            code.crz(lamb, entangle_line_idx, idx)
        elif has_entangle == False:
            code.rz(lamb, idx)

    def strs(self):
        return "|RZ|"

class Inverse_QFT(object):
    def __init__(self):
        self.type = Inverse_QFT

    def strs(self):
        return "|IQ|"

class U_Gate(object):
    def __init__(self, theta, phi, lamb):
        self.type = "U"
        self.theta = theta
        self.phi = phi
        self.lamb = lamb
        self.has_entangle = True

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        if has_entangle == True:
            code.u(self.theta, self.phi, self.lamb, 0, entangle_line_idx, idx)
        elif has_entangle == False:
            code.u(self.theta, self.phi, self.lamb, idx)

    def strs(self):
        return "|U |"

class CNOT_Gate(object):
    def __init__(self):
        self.type = CNOT_Gate
        self.has_entangle = True
        self.has_parameter = True

    def add_code(self, code, idx, has_entangle=False, entangle_line_idx=0):
        code.cx(entangle_line_idx, idx)

    def strs(self):
        return "|N |"

class controled_CNOT_Gate(object):
    def __init__(self):
        self.type = controled_CNOT_Gate
        self.has_entangle = False

    def strs(self):
        return "|x |"

class Controled_Point(object):
    def __init__(self):
        self.type = Controled_Point

    def add_code(self, code, idx):
        return

    def strs(self):
        return "|x |"

class Empty_Gate(object):
    def __init__(self):
        self.type = None

    def add_code(self, code, idx):
        code.barrier(idx)

    def strs(self):
        return "|==|"

class Classical_Line(object):
    def __init__(self):
        self.type = "Classical"

    def strs(self):
        return ""

class Line(object):
    def __init__(self):
        self.route = list()
        self.initial_value = 0
        pass

    def add_control_not_gate(self):
        self.route.append(CNOT_Gate())

    def add_controled_not_gate(self):
        self.route.append(controled_CNOT_Gate())
    
    def add_u_gate(self, theta, phi, lamb):
        self.route.append(U_Gate(theta, phi, lamb))

    def add_empty_gate(self):
        self.route.append(Empty_Gate())

    def add_gate(self, gate):
        self.route.append(gate)

    def __len__(self):
        return len(self.route)

    def __str__(self):
        output = "=|0>==="
        for r in self.route:
            output = output + r.strs()
            output = output + "=="
        return output

class Circuit(object):
    def __init__(self, num_qubits):
        self.lines = list()
        self.num_qubits = num_qubits
        self.num_cbits = num_qubits
        self.last_modified_line_len = 0
        for i in range(num_qubits):
            line = Line()
            self.lines.append(line)
        self.code = self.initial_seeds(self.num_qubits, self.num_cbits)
        self.results = gen_bin_dict(self.num_qubits)

    def get_longgest_length(self):
        len_max = max([len(x) for x in self.lines])
        return len_max

    def _add_superposition(self):
        hardmard_gate = H_Gate()
        for i in range(len(self.lines)):
            self.lines[i].add_gate(hardmard_gate)
            hardmard_gate.add_code(code=self.code, idx=i)

    def __str__(self):
        strs = ""
        for i, l in enumerate(self.lines):
            strs = strs + str(l)
            if i == len(self.lines) - 1:
                return strs
            else:
                strs = strs + "\n"

    def add_QFT(self, qft):
        self.code.append(qft, list(range(self.num_qubits)))
        for line in self.lines:
            line.route.append(Inverse_QFT())

    def run_code(self):
        for i in range(self.num_qubits):
            self.code.measure(i, i)
        result = execute(self.code, Aer.get_backend('aer_simulator'), shots=10000).result().get_counts(self.code)
        for k, v in result.items():
            self.results[k] = v
        return self.results

    def run_vec(self):
        result = execute(self.code, Aer.get_backend('statevector_simulator'), shots=10000).result().get_statevector(self.code)
        return result

    def add_mcgate(self, idx_list, target_line):
        self.code.mcx(idx_list, target_line, mode='noancilla')
        for i in idx_list:
            self.lines[i].add_controled_not_gate()
        self.lines[target_line].add_gate(X_Gate())

    def add_ccx_gate(self, line_idx, first_control, second_control):
        self.lines[line_idx].add_control_not_gate()
        self.lines[first_control].add_controled_not_gate()
        self.lines[second_control].add_controled_not_gate()

    def add_gate(self, line_idx, gate, has_entangle=False, entangle_line_idx=0):
        self.lines[line_idx].add_gate(gate)
        self.last_modified_line_len = self.get_longgest_length()
        class_name = gate.__class__.__name__
        modules = __import__("line")
        get_class = getattr(modules, class_name)
        obj = get_class()
        if has_entangle == True:
            self.add_controled_point(entangle_line_idx)
            obj.add_code(code=self.code, idx=line_idx, has_entangle=True, entangle_line_idx=entangle_line_idx)
        elif has_entangle == False:
            obj.add_code(code=self.code, idx=line_idx)

    def add_u_gate(self, line_idx, rotation, has_entangle=False, entangle_line_idx=0):
        theta, fi, lamb = rotation
        self.last_modified_line_len = self.get_longgest_length()
        u_gate = U_Gate(theta, fi, lamb)
        self.lines[line_idx].add_gate(u_gate)
        if has_entangle == True:
            u_gate.add_code(code=self.code, idx=line_idx, has_entangle=True, entangle_line_idx=entangle_line_idx)
        elif has_entangle == False:
            u_gate.add_code(code=self.code, idx=line_idx)

    def add_controled_point(self, line_idx):
        controled_point = Controled_Point()
        self.add_gate(line_idx, controled_point)
        self.last_modified_line_len = len(self.lines[line_idx])

    def fill_with_empty_gate(self):
        for i in range(self.num_qubits):
            if len(self.lines[i]) == self.last_modified_line_len:
                continue
            else:
                empty = Empty_Gate()
                empty.add_code(self.code, i)
                self.lines[i].add_empty_gate()

    def initial_seeds(self, num_qubits, num_cbits):
        q = QuantumRegister(num_qubits)
        c = ClassicalRegister(num_cbits)
        qc = QuantumCircuit(q, c)
        return qc

    def output_existing_entanglement(self):
        existing_entanglement = list()
        for line_idx, line in enumerate(self.lines):
            entanglement = [1] * len(self.lines)
            for idx, gate in enumerate(line.route):
                if not isinstance(gate, control_CNOT_Gate):
                    continue
                entanglement[line_idx] = 0
                for ll_idx, ll in enumerate(self.lines):
                    if ll == self.lines[line_idx]:
                        continue
                    if not isinstance(ll[idx], controled_CNOT_Gate):
                        continue
                    else:
                        entanglement[ll_idx] = 1 
            existing_entanglement.append(tuple(entanglement))
        return existing_entanglement

    def output_existing_rotation(self):
        existing_rotation = list()
        for line in self.lines:
            for gate in line.route:
                if isinstance(gate, U_Gate):
                    theta, phi, lamb = gate.theta, gate.phi, gate.lamb
                    existing_rotation.append((theta, phi, lamb))
        return existing_rotation

if __name__ == "__main__":
    circuit = Circuit(3)
    circuit.add_gate(0, H_Gate())
    circuit.add_gate(1, CNOT_Gate(), has_entangle = True, entangle_line_idx = 0)
    circuit.add_ccx_gate(0,1,2)
    print(circuit)
    circuit.run_code()
    print(circuit.results)

    