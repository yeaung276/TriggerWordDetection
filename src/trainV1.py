import sys
import numpy as np
from keras.optimizers import Adam
from Utils.td_utils import Logger
from Utils.train_utils import TrainingExamplesGenerator
from configs.tr_config import Ty, tr_input_shape
from Net_Architectures.tr_model import create_tr_model


# generating data
X = None
Y = None
def generate_data():
    global X
    global Y
   
    Generator = TrainingExamplesGenerator(
        log=True, 
        seed=10
    )
    Generator.load_data(path='raw_data')
    X, Y = Generator.generate_examples(2000)
    np.save('train_X.npy', X)
    np.save('train_Y.npy', Y)

# Training Data
def Train():
  global X
  global Y

  X = np.load('training_data/X_train.npy')
  Y = np.load('training_data/Y_train.npy')
  
  model = create_tr_model(tr_input_shape)
  opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
  model.fit(X, Y, batch_size = 5, epochs=100)
  model.save('Models/model1.h5')

sys.stdout = Logger('trainV1_log.txt')
# generate_data()
Train()