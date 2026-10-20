import os
import sys
from load_benchmark import load_benchmark
from mutator import RandomMutator, QFTMutator, UCNOTMutator
from line import Circuit
import re
from logzero import logger
import termplotlib as tpl
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)

mutators = {
    "random": RandomMutator(),
    "qft": QFTMutator(),
    "ucnot": UCNOTMutator()
}

benchmark_path_list = load_benchmark()

def get_num_qubits(file_path):
    return int(re.findall("[0-9]+", file_path)[0])

def init_circuit(mutator_name, num_qubits):
    mutator = mutators[mutator_name]
    init_circuit = Circuit(num_qubits)
    new_circuit = mutator.generate_circuit(init_circuit)
    return new_circuit

def compose_circuit(input_circuit, file_path, num_qubits):
    current_benchmark = QuantumCircuit.from_qasm_file(file_path)
    composed_circuit = input_circuit.code.compose(current_benchmark)
    for i in range(num_qubits):
        composed_circuit.measure(i, i)
    return composed_circuit

def run_code(circuit):
    results = execute(circuit, Aer.get_backend('aer_simulator'), shots=100000).result().get_counts(circuit)
    return results

def main(mutator_name, run_idx):
    for file_path in benchmark_path_list:
        num_qubits = get_num_qubits(file_path)
        if "pe_" in file_path:
            num_qubits += 1
        if num_qubits > 10 or num_qubits < 4:
            continue
        if "dynamic" in file_path and num_qubits > 15:
            continue
        logger.debug("Try to run mutator: {0} at {1}".format(mutator_name, file_path))
        new_circuit = init_circuit(mutator_name, num_qubits)
        composed_circuit = compose_circuit(new_circuit, file_path, num_qubits)
        results = run_code(composed_circuit)
        head_folder = "exp-04" + "-" + str(run_idx).zfill(2)
        if os.path.exists(head_folder) == False:
            os.mkdir(head_folder)
        save_path = os.path.join(head_folder, mutator_name)
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)
        file_name = mutator_name + "_" + file_path.split('/')[-1].split('.')[0]
        final_save_path = os.path.join(save_path, file_name)
        with open(final_save_path + "_result", 'w') as f:
            print(results, file = f)
        with open(final_save_path + "_circuit", 'w') as f:
            print(composed_circuit.qasm(), file = f)


def main_for_baseline(mutator_name, run_idx):
    return main(mutator_name, run_idx)

if __name__ == "__main__":
    for i in range(50):
        for mutator_name in mutators.keys():
            main(mutator_name, i)