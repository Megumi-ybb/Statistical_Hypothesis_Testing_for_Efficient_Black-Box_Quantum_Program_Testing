import os
import sys
from logzero import logger
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from qiskit.circuit.library import QFT
import secrets
from copy import deepcopy
from line import gen_bin_dict
from line import Circuit
from math import gcd,log
import math
import random
import numpy as np
from qiskit.circuit.library.phase_oracle import PhaseOracle
from sat import Generator
from z3 import *
import termplotlib as tpl

def generate_n_random_choice(ls, num):
    s = set()
    while len(s) < num:
        ch = secrets.choice(ls)
        if ch not in s:
            s.add(ch)
    return list(s)

class FullAdder(object):
    def __init__(self, qc, num_qubits):
        self.qc = qc.code
        self.num_qubits = num_qubits
        self.results = None
        self.idxs = None
        self.threshold = 0.01
        self.algorithm()

    def algorithm(self):
        choices = generate_n_random_choice(list(range(self.num_qubits)), 4)
        A, B, sum_out, carry_out = choices
        self.idxs = choices
        self._process(self.qc, A, B, sum_out, carry_out)
        self._run()

    def _process(self, qc, A, B, sum_out, carry_out):
        qc.ccx(A, B, carry_out)
        qc.cx(A, B)
        qc.ccx(B, sum_out, carry_out)
        qc.cx(B, sum_out)
        qc.cx(A, B)

    def adder(self, a, b, c, d):
        ssum = c ^ (a ^ b)
        c_out = b&(not a^b)|(not a )&b
        c_out = c_out ^ d
        return c_out, ssum, b, a

    def _run(self):
        for i in range(self.num_qubits):
            self.qc.measure(i, i)
        self.results = execute(self.qc, Aer.get_backend('aer_simulator'), shots=10000).result().get_counts(self.qc)

    def change_freq_into_prob(self, distribution):
        for value, freq in distribution.items():
            tmp = freq / 10000
            distribution[value] = tmp
        return distribution


    def result_verify(self, input_distribution):
        print(self.qc)
        output_dict = gen_bin_dict(self.num_qubits)
        for k, v in self.results.items():
            output_dict[k] = v
        input_probs = self.change_freq_into_prob(input_distribution)
        output_probs = self.change_freq_into_prob(output_dict)
        A, B, sum_out, carry_out = self.idxs
        for value in input_probs.keys():
            if input_probs[value] == 0.0:
                continue
            value_copy = deepcopy(value)
            input_prob = input_probs[value]
            p_a = int(value[::-1][A])
            p_b = int(value[::-1][B])
            p_c = int(value[::-1][sum_out])
            p_d = int(value[::-1][carry_out])
            pp_d, pp_c, pp_b, pp_a = self.adder(p_a, p_b, p_c, p_d)
            pp_d, pp_c, pp_b, pp_a = str(pp_d), str(pp_c), str(pp_b), str(pp_a)
            list_tmp = list()
            for i in value_copy:
                list_tmp.append(i)
            list_tmp[A] = pp_a
            list_tmp[B] = pp_b
            list_tmp[sum_out] = pp_c
            list_tmp[carry_out] = pp_d
            list_tmp.reverse()
            value_copy = "".join(list_tmp)
            try:
                output_prob = output_probs[value_copy]
            except:
                logger.debug("input_result: {}".format(input_distribution))
                logger.debug("output_result: {}".format(self.results))
                break
            distance = abs(output_prob - input_prob)
            if distance > self.threshold:
                logger.warning("From {0} to {1} Incorrect Calculation!".format(value, value_copy))
                logger.warning("Expected {0} But Got {1}".format(output_prob, input_prob))
                return False
        logger.debug("All Results Correct!")
        return True

class QuantumPhaseEstimation(object):
    def __init__(self, qc, num_qubits):
        self.qc = qc.code
        self.num_qubits = num_qubits
        self.p = 1
        self.n = self.num_qubits - self.p
        self.results = None
        self.qpe(self.qc)
        self._run()

    def _add_measurement(self, qc):
        for i in range(self.num_qubits - 1):
            self.qc.measure(i, i)

    def _run(self):
        for i in range(self.num_qubits - 1):
            self.qc.measure(i, i)
        self.results = execute(self.qc, Aer.get_backend('aer_simulator'), shots=10000).result().get_counts(self.qc)

    def _process(self, qc):
        angle = math.pi/6
        for i in range(self.n):
            qc.h(i)
        for i in range(self.n, self.num_qubits):
            qc.x(i)
        line_idx = 0
        while line_idx < self.num_qubits - 1:
            num_gates = pow(2, line_idx)
            last_idx = self.num_qubits - 1
            for _ in range(num_gates):
                qc.cp(angle, line_idx, last_idx)
            line_idx += 1

    def _add_qft(self, qc):
        qft = QFT(num_qubits = self.num_qubits - 1, approximation_degree = 0, \
             do_swaps = True, inverse = True, insert_barriers = True, name = "qft")
        qc.append(qft, list(range(self.num_qubits-1)))

    def qpe(self, qc):
        self._process(self.qc)
        self._add_qft(self.qc)
        self._add_measurement(self.qc)

    def result_verify(self, results):
        pass

class Grover(object):
    def __init__(self, qc, num_qubits, instance=None, load_inst_from_file=None):
        self.qc = qc.code
        self.num_qubits = num_qubits
        self.results = None
        self.idxs = None
        self.threshold = 0.01
        self.load_inst_from_file = load_inst_from_file
        self.generator = Generator(4, self.num_qubits)
        if instance != None:
            self.instance = instance
        else:
            self.instance = self.generator.instance
        self.result_of_grover = None
        self.grover()
        self._run()
        
    def _run(self):
        for i in range(self.num_qubits):
            self.qc.measure(i, i)
        self.results = execute(self.qc, Aer.get_backend('aer_simulator'), shots=10000).result().get_counts(self.qc)

    def grover(self):
        instance = self.instance
        if self.load_inst_from_file == None:
            oracle = PhaseOracle(self.generator.strs)
        else:
            oracle = QuantumCircuit.from_qasm_file(self.load_inst_from_file)
        print(oracle)
        self.qc = self.qc.compose(oracle)
        self._amplification(self.qc)

    def digital_to_boolean(self, digital):
        bools = list()
        for s in digital:
            if s == '1':
                bools.append(True)
            elif s == '0':
                bools.append(False)
        return bools

    def result_verify(self):
        bit_value, freq = list(self.results.keys()), list(self.results.values())
        figure = tpl.figure()
        figure.barh(freq, bit_value, show_vals = False)
        sorted_idx = sorted(self.results, reverse = True, key=lambda x: self.results[x])
        result_list = self.digital_to_boolean(sorted_idx[0])
        self.result_of_grover = sorted_idx[0]
        var_list = [i for i in "abcdefghijklmnopqrstuvwxyz"]
        var_str = " ".join(var_list[:self.num_qubits])
        bool_list = Bools(var_str)
        solver = Solver()
        solver.add(self.instance)
        for i in range(self.num_qubits):
            if result_list[i] == True:
                solver.add(bool_list[i] == True)
            else:
                solver.add(bool_list[i] == False)
        if solver.check() == sat:
            return True
            print("The answer {} is verified.".format(sorted_idx[0]))
        else:
            return False
            print("The answer {} is incorrect.".format(sorted_idx[0]))

    def _amplification(self, qc):
        for i in range(self.num_qubits):
            self.qc.h(i)
            self.qc.x(i)
        self.qc.h(self.num_qubits-1)
        self.qc.mct([i for i in range(self.num_qubits -1)], self.num_qubits-1)
        self.qc.h(self.num_qubits-1)
        for i in range(self.num_qubits):
            self.qc.x(i)
            self.qc.h(i)

if __name__ == "__main__":
    num_qubits = 3
    cirq = Circuit(num_qubits)
    oracle = Grover(cirq, num_qubits)
