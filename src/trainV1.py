import sys
import numpy as np
from Utils.td_utils import Logger
from Utils.train_utils import TrainingExamplesGenerator
from configs.tr_config import Ty

# generating data
def generate_data():
    sys.stdout = Logger('trainV1_log.txt')
    Generator = TrainingExamplesGenerator(
        log=True, 
        seed=10
    )
    Generator.load_data(path='raw_data')
    X, Y = Generator.generate_examples(2)
    np.save('train_X.npy', X)
    np.save('train_Y.npy', Y)

generate_data()