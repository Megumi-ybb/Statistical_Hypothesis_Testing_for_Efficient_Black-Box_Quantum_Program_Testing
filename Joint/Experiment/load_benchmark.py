import os
import inspect
import sys

BENCHMARK_FOLDER = "benchmark"
CATEGORIES = [
    "combinational",
    "dynamic",
]
QUBIT_LIMIT = 10

def get_classes():
    classes = list()
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for (name, _) in clsmembers:
        if "Load" in name:
            classes.append(name)
    return classes

def load_benchmark():
    load_list = list()
    classes = get_classes()
    for cls in classes:
        modules = __import__("load_benchmark")
        get_class = getattr(modules, cls)
        obj = get_class()
        if obj.category in CATEGORIES:
            load_list += obj.file_list
    return load_list

class LoadAdder(object):
    def __init__(self):
        self.category = "combinational"
        self.folder_name = "adder"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list

class LoadBV(object):
    def __init__(self):
        self.category = "combinational"
        self.folder_name = "bv"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list


class LoadGrover(object):
    def __init__(self):
        self.category = "combinational"
        self.folder_name = "grover"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list

class LoadPE(object):
    def __init__(self):
        self.category = "combinational"
        self.folder_name = "pe"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list

class LoadQFT(object):
    def __init__(self):
        self.category = "combinational"
        self.folder_name = "qft"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list

class LoadRC(object):
    def __init__(self):
        self.category = "combinational"
        self.folder_name = "rand_cliff"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list

class LoadDPE(object):
    def __init__(self):
        self.category = "dynamic"
        self.folder_name = "pe"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list

class LoadDQFT(object):
    def __init__(self):
        self.category = "dynamic"
        self.folder_name = "qft"
        self.folder_path = os.path.join(BENCHMARK_FOLDER, self.category, self.folder_name)
        self.file_list = self.load_files(self.folder_path)

    def load_files(self, path):
        file_list = list()
        for ff in os.listdir(self.folder_path):
            file_list.append(os.path.join(path, ff))
        return file_list


if __name__ == "__main__":
    load_benchmark()
    print(get_classes())