import sys
import numpy as np
import keras
import scipy.io
from keras.optimizers import Adam
from Utils.td_utils import Logger
from Utils.train_utils import TrainingExamplesGenerator
from configs.tr_config import Ty, tr_input_shape
from Net_Architectures.tr_model import create_tr_model

#hypermeters
EXAMPLES = 2000
BATCH_SIZE = 5
LEARNING_RATE = 0.0001
BETA_1 = 0.9
BETA_2 = 0.999
DECAY_RATE = 0.01
EPOCHS = 10

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
    for X, Y in Generator.generate_examples(2000):
      np.save('train_X.npy', X)
      np.save('train_Y.npy', Y)

# Training Data

Generator = TrainingExamplesGenerator(
      log=False, 
      seed=10
  )
Generator.load_data(path='raw_data')
gen = Generator.generate_examples(EXAMPLES, batch_size=BATCH_SIZE)

model = create_tr_model(tr_input_shape)
opt = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, decay=DECAY_RATE)

recall = keras.metrics.Recall()
precission = keras.metrics.Precision()

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[precission, recall])
history = model.fit(gen, steps_per_epoch=2000/5, epochs=EPOCHS)
model.save('Models/model1.h5')
scipy.io.savemat('Models/model1_train_history.mat', history)
