import os
import sys
from logzero import logger
from line import Circuit, CNOT_Gate, H_Gate, X_Gate
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from qiskit.circuit.library import QFT
from oracle import Grover
from mutator import RandomMutator, QFTMutator, UCNOTMutator
from copy import deepcopy
from analysis_generator2 import phase_analysis, entangle_analysis, mag_analysis, write_to_csv, get_high_phase_score_detail
from grid_score import Gridstatistic
import time, copy
import termplotlib as tpl
import numpy as np
import random
from sat import Generator
from datetime import datetime
from run_benchmark import main_for_baseline


CONFIG = {
    "circuit_path": "circuit.py",
    "budget": 1500,
    "num_input_qubit": 1,
    "num_classical_bit": 1,
}



from qiskit.quantum_info import Statevector

def qubit_marginals(vec, num_qubits):
    sv = Statevector(vec)
    probs = []
    for i in range(num_qubits):
        p0 = sv.probabilities([i])[0]
        p1 = 1 - p0
        probs.append((p0, p1))
    return probs


class StormFuzzer(object):
    def __init__(self, num_qubits = 5, seed_num = 100, test_num=5, test_case_size = 15):
        self.num_qubits = num_qubits
        self.seed_num = seed_num
        self.test_num = test_num
        self.test_case_size = test_case_size
        self.seeds = []

        self.mean_min, self.mean_max = -15, 15
        self.var_min, self.var_max = 0.1, 30

    def initialize(self, verbose=True):
        self.seeds = [
            (np.random.uniform(self.mean_min, self.mean_max),
             np.random.uniform(self.var_min, self.var_max))
            for _ in range(self.seed_num)
        ]
        if verbose:
            print(f"Initialized {len(self.seeds)} seed distributions.")


    def generate(self, seed):
        test_case_list = []
        mean = seed[0]
        vars = seed[1]
        for _ in range(self.test_num):
            circuit = Circuit(self.num_qubits)
            sample = np.random.normal(mean, vars, self.test_case_size)
            for i in range(self.num_qubits):
                circuit.add_u_gate(i, (sample[i * 3], sample[i * 3 + 1], sample[i * 3 + 2]))
            qft = QFT(num_qubits = self.num_qubits, approximation_degree = 0, do_swaps = True, inverse = True, insert_barriers = False, name = "qft")
            circuit.add_QFT(qft)
            
            test_case_list.append(copy.deepcopy(circuit))
        return test_case_list

    def mutate(self):
        new_seeds = []

        for seed in self.seeds:
            mean = seed[0]
            vars = seed[1] 
            new_mean = np.clip(mean + np.random.uniform(-0.5,0.5), self.mean_min, self.mean_max)
            new_vars = np.clip(vars + np.random.uniform(-0.5,0.5), self.var_min, self.var_max)

            if np.random.uniform(0,1) < 0.1:
                new_mean = np.random.uniform(self.mean_min, self.mean_max)
                new_vars = np.random.uniform(self.var_min, self.var_max)

            new_seeds.append((new_mean, new_vars))
        self.seeds = self.seeds + new_seeds
            
    def select(self, metrics):
        sorted_seeds = [x for _, x in sorted(zip(metrics, self.seeds), reverse=True)]
        self.seeds = sorted_seeds[:self.seed_num]
 

class Prometheus(object):
    def __init__(self, mutator, num_qubits):
        self.num_qubits = int(num_qubits)
        self.mutator = mutator
        self.circuit = Circuit(self.num_qubits)
        self.current_count = 0
        self.available_oracles = ['03', '05', '07', '12', '13', '16', '24', '26']
        self.grids = Gridstatistic(score_num=3, grid_size= 0.1)
        self.generator = Generator(4, num_qubits)
        self.instance = self.generator.constant_instance()
        self.bug_counter = 0

        self.stats = [[],[],[]]

    def testing(self):

        seed_metric_list = []
        for seed in self.mutator.seeds:
            new_circuit_list = self.mutator.generate(seed)
            new_circuit_metric_list = []
            for new_circuit in new_circuit_list:
                oracle = random.choice(self.available_oracles)
                copy_new_circuit = deepcopy(new_circuit)
                vec_circuit = deepcopy(new_circuit)
                result = new_circuit.run_code()
                phase_result = vec_circuit.run_vec()

                marginals = qubit_marginals(phase_result, self.num_qubits)
                print("每个qubit的(P(0),P(1))：", marginals)
                print("STATE VECTOR = ", phase_result)
                mag, phase, entan = mag_analysis(num_qubits, result), phase_analysis(num_qubits, phase_result), entangle_analysis(result, num_qubits)
                a, b, c = mag[:num_qubits],phase[:num_qubits],entan[:num_qubits]
                self.stats[0].append(a)
                self.stats[1].append(b)
                self.stats[2].append(c)
                sample_id, occureence = self.grids.cluster(self.num_qubits, result, phase_result)
                novelty = self.grids.cal_novelty(occureence)
                self.current_count += 1
                avg_novelty = np.mean(novelty)
                new_circuit_metric_list.append(avg_novelty)
                print(new_circuit)

                grover = Grover(new_circuit, num_qubits, self.instance, load_inst_from_file='inst/' + oracle + '.or')
                verified = grover.result_verify()

                if verified:
                    pass
                else:
                    self.bug_counter += 1



            avg_seed_metric = np.mean(new_circuit_metric_list)
            seed_metric_list.append(avg_seed_metric)
        return seed_metric_list
        

    def start_testing(self):
        print("✅ 进入 start_testing")
        budget = CONFIG['budget']

        self.mutator.initialize()
        print("✅ mutator 初始化完成")
        
        round_count = 1

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"fuzz_log_{timestamp}.txt")

        print(f"📁 当前运行日志文件: {log_file}")

        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("轮数\t总测试数\t非零grdis\tbugs\tmean\tvars\n")

        while self.current_count <= budget:
            print("当前轮:", round_count, "当前总测试数:", self.current_count)
            self.mutator.mutate()
            seed_metric_list = self.testing()
            self.mutator.select(seed_metric_list)
            grdis_count = np.count_nonzero(self.grids.counter)
            print("current_count：", self.current_count)

            with open(log_file, 'a', encoding='utf-8') as f:
                for mean, vars in self.mutator.seeds:
                    f.write(f"{round_count}\t{self.current_count}\t{grdis_count}\t{self.bug_counter}\t{mean}\t{vars}\t{datetime.now()}\t{num_qubits}\n")

            if self.current_count % 100 == 0:
                logger.debug("Now on {} course".format(self.current_count))
                logger.debug("The tester covered all {0} grdis!".format(np.count_nonzero(self.grids.counter)))        
                logger.debug("The tester covered all {0} bugs!".format(self.bug_counter))

            print(f"✅ Round {round_count} finished. current_count: {self.current_count}, grdis: {grdis_count}, bugs: {self.bug_counter}")
            round_count += 1



    def draw3d(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('mag')
        ax.set_ylabel('phase')
        ax.set_zlabel('entan')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])


        ax.scatter(self.stats[0], self.stats[1], self.stats[2], c='blue', marker='o')

        plt.show()

    def draw2d(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.ticker import MultipleLocator

        mags = np.array(self.stats[0]).flatten()
        phases = np.array(self.stats[1]).flatten()
        entans = np.array(self.stats[2]).flatten()

        os.makedirs("results", exist_ok=True)
        print("保存目录路径为：", os.path.abspath("results"))
        np.save("results/mag.npy", mags)
        np.save("results/phase.npy", phases)
        np.save("results/entan.npy", entans)
        print("✅ 数据已保存至 results/ 目录下的 mag.npy, phase.npy, entan.npy")

        tick_interval = 0.1
        locator = MultipleLocator(tick_interval)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(mags, phases, c='b', s=10)
        axes[0].set_xlabel('magnitude')
        axes[0].set_ylabel('phase')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].xaxis.set_major_locator(locator)
        axes[0].yaxis.set_major_locator(locator)
        axes[0].tick_params(axis='both', which='both', direction='in', length=5)
        axes[0].set_title('magnitude vs phase')

        axes[1].scatter(mags, entans, c='g', s=10)
        axes[1].set_xlabel('magnitude')
        axes[1].set_ylabel('entanglement')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].xaxis.set_major_locator(locator)
        axes[1].yaxis.set_major_locator(locator)
        axes[1].tick_params(axis='both', which='both', direction='in', length=5)
        axes[1].set_title('magnitude vs entanglement')

        axes[2].scatter(phases, entans, c='r', s=10)
        axes[2].set_xlabel('phase')
        axes[2].set_ylabel('entanglement')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].xaxis.set_major_locator(locator)
        axes[2].yaxis.set_major_locator(locator)
        axes[2].tick_params(axis='both', which='both', direction='in', length=5)
        axes[2].set_title('phase vs entanglement')

        plt.tight_layout()
        plt.show()
            
if __name__ == '__main__':
    num_qubits = 5
    mutator = StormFuzzer(num_qubits = num_qubits, seed_num=10, test_num = 1, test_case_size= num_qubits * 3)
    p = Prometheus(mutator, num_qubits)
    p.start_testing()
    p.draw3d()
    p.draw2d()

    
def produce_state(num_qubits: int, seed: int = None):
    from copy import deepcopy
    import numpy as np

    sf = StormFuzzer(num_qubits=num_qubits, seed_num=1, test_num=1, test_case_size=num_qubits * 3)
    sf.initialize(verbose=False)
    seed_pair = sf.seeds[0]

    circuits = sf.generate(seed_pair)
    circuit = circuits[0]

    vec = deepcopy(circuit).run_vec()

    norm = np.linalg.norm(vec)
    if norm > 0 and not np.isclose(norm, 1.0, atol=1e-9):
        vec = vec / norm

    return vec