import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import Adam

from configs.tr_config import tr_input_shape
from configs.tr_config import nfft, fs, noverlap
from Net_Architectures.tr_model import create_model
from Utils.td_utils import graph_spectrogram, get_wav_info


class TriggerWordDetector:
    def __init__(self, model:str):
        self.model = create_model()
        # self.model.load_weights(model)

    def predict_on_wav_file(self, filename: str):
        """
        used for predecting on a wav file
        params: filename: filename of the wave file to predict, should be 10 seconds long
        return: prediction sequencs of length Ty
        """
        x = graph_spectrogram(filename)
        x  = x.swapaxes(0,1)
        assert x.shape == tr_input_shape, 'invalid input shape. Expected {},' \
            'but get{}'.format(tr_input_shape,x.shape)
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)
        return predictions

    def get_spectrogram(self, data):
        """
        compute the seectrogram of the datasequence
        params: data: time domain data sequence 
        return: spectrogram of the input data with shape (Tx, freqs)
        """
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        # the spectrogram outputs (freqs, Tx) require outputs (Tx, freqs)
        return pxx.swapaxes(0,1)

    def test(self):
        self.get_spectrogram('Audios/example_train.wav')

# model = load_model('Models/tr_model.h5')
# model = create_model()
# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
# model.load_weights('Models/tr_model.h5', skip_mismatch=True)
# model.summary()
d = TriggerWordDetector(model='Models/tr_model.h5')
d.test()
