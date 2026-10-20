import numpy as np
from analysis_generator2 import phase_analysis, entangle_analysis, mag_analysis, write_to_csv
import math

class Gridstatistic():
    def __init__(self, score_num, grid_size):
        self.dim = score_num
        self.step = grid_size
        self.counter = self.generate_count_grid()
        self.counter1 = self.generate_count_grid()
        self.fake_counter = self.generate_count_grid()

        self.statis = [[], [], []]
        self.fake_statis = [[], [], []]

    def generate_count_grid(self):
        shape = (int(1/self.step),) * self.dim
        matrix = np.zeros(shape)
        return matrix

    def update_counter(self, score_list):
        count_index = []
        for i in range(self.dim):
            count_index.append(self.find_interval(score_list[i]))
        self.counter[eval(str(count_index).replace("[", "").replace("]", ""))] += 1
        return count_index, self.counter[eval(str(count_index).replace("[", "").replace("]", ""))]
    

    def fake_update_counter(self, score_list):
        count_index = []
        for i in range(self.dim):
            count_index.append(self.find_interval(score_list[i]))
        return count_index, self.counter[eval(str(count_index).replace("[", "").replace("]", ""))]
    


    def fake_update_counter1(self, score_list):
        count_index = []
        for i in range(self.dim):
            count_index.append(self.find_interval(score_list[i]))
        self.fake_counter[eval(str(count_index).replace("[", "").replace("]", ""))] += 1
        return count_index, self.fake_counter[eval(str(count_index).replace("[", "").replace("]", ""))]



    def update_counter1(self, score_list):
        count_index = []
        for i in range(self.dim):
            count_index.append(self.find_interval(score_list[i]/1000))
        self.counter1[eval(str(count_index).replace("[", "").replace("]", ""))] += 1
        return count_index, self.counter1[eval(str(count_index).replace("[", "").replace("]", ""))]


    def find_interval(self, number):
        points = [i * self.step for i in range(int(1 / self.step) + 1)]

        for i in range(len(points) - 1):
            if points[i] <= number < points[i + 1]:
                return i

        if number == 1:
            return len(points) - 2

        print("strange score!")
        return None


    def novelty(self, count):
        if count > 20:
            count = 20
        return 1/np.exp(count)

    def cal_all_novelty(self):
        print("The tester covered all {0} grdis!".format(np.count_nonzero(self.counter)))
        novelty_matrix = self.novelty(self.counter)
        return np.sum(novelty_matrix)/self.counter.size
    
    def cluster(self, num_qubits, result, phase_result):
        mag, phase, entan = mag_analysis(num_qubits, result), phase_analysis(num_qubits, phase_result), entangle_analysis(result, num_qubits)
        self.statis[0].extend(mag)
        self.statis[1].extend(phase)
        self.statis[2].extend(entan)


        index, occurance = [], []
        for i in range(num_qubits):
            idx, occur = self.update_counter([mag[i], phase[i], entan[i]])
            index.append(idx)
            occurance.append(occur)
        return index, occurance
    
    def fake_cluster(self, num_qubits, result, phase_result):
        mag, phase, entan = mag_analysis(num_qubits, result), phase_analysis(num_qubits, phase_result), entangle_analysis(result, num_qubits)
        index, occurance = [], []
        for i in range(num_qubits):
            idx, occur = self.fake_update_counter([mag[i], phase[i], entan[i]])
            index.append(idx)
            occurance.append(occur)
        return index, occurance
    

    def fake_cluster1(self, num_qubits, result, phase_result):
        mag, phase, entan = mag_analysis(num_qubits, result), phase_analysis(num_qubits, phase_result), entangle_analysis(result, num_qubits)
        self.fake_statis[0].extend(mag)
        self.fake_statis[1].extend(phase)
        self.fake_statis[2].extend(entan)
        index, occurance = [], []
        for i in range(num_qubits):
            idx, occur = self.fake_update_counter1([mag[i], phase[i], entan[i]])
            index.append(idx)
            occurance.append(occur)
        return index, occurance
    


        
        
    
    
    def cal_novelty(self, occurance):
        print("The tester covered all {0} grdis!".format(np.count_nonzero(self.counter)))
        novelties = [self.novelty(i) for i in occurance]
        return novelties
    
    def fake_cal_novelty(self, occurance):
        novelties = [self.novelty(i) for i in occurance]
        return novelties
        



if __name__ == "__main__":
    g = Gridstatistic(3, 0.1)
    grid = g.generate_count_grid()
    for i in range(100):
        g.update_counter([0.1, 0.2, 0.2])
        print(g.cal_all_novelty())

    for i in range(100):
        g.update_counter([0.1, 0.7, 0.2])
        print(g.cal_all_novelty())




