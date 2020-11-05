# import numpy as np
# import time
# import pyaudio
# from queue import Queue
# from threading import Thread
# import sys
# import time

# chunk_duration = 0.5 # Each read length in seconds from mic.
# fs = 44100 # sampling rate for mic
# chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# # Each model input data duration in seconds, need to be an integer numbers of chunk_duration
# feed_duration = 10
# feed_samples = int(fs * feed_duration)

# assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

# def get_audio_input_stream(callback):
#     stream = pyaudio.PyAudio().open(
#         format=pyaudio.paInt16,
#         channels=1,
#         rate=fs,
#         input=True,
#         frames_per_buffer=chunk_samples,
#         # input_device_index=0,
#         stream_callback=callback)
#     return stream

# q = Queue()

# run = True

# silence_threshold = 1000

# # Run the demo for a timeout seconds
# timeout = time.time() + 0.5*60  # 0.5 minutes from now

# # Data buffer for the input wavform
# data = np.zeros(feed_samples, dtype='int16')

# def callback(in_data, frame_count, time_info, status):
#     global run, timeout, data, silence_threshold    
#     if time.time() > timeout:
#         run = False        
#     data0 = np.frombuffer(in_data, dtype='int16')
#     print(np.abs(data0).mean())
#     if np.abs(data0).mean() < silence_threshold:
#         sys.stdout.write('-')
#         return (in_data, pyaudio.paContinue)
#     else:
#         sys.stdout.write('.')
#     data = np.append(data,data0)    
#     if len(data) > feed_samples:
#         data = data[-feed_samples:]
#         # Process data async by sending a queue.
#         q.put(data)
#     return (in_data, pyaudio.paContinue)

# stream = get_audio_input_stream(callback)
# stream.start_stream()

# # p = pyaudio.PyAudio()
# # for i in range(p.get_device_count()):
# #   dev = p.get_device_info_by_index(i)
# #   print((i,dev['name'],dev['maxInputChannels']))


# # import pyaudio
# try:
#     while run:
#         data = q.get()
#         # spectrum = get_spectrogram(data)
#         # preds = detect_triggerword_spectrum(spectrum)
#         # new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
#         # if new_trigger:
#         # sys.stdout.write('1')
#         # print(np.abs(data).mean())
# except (KeyboardInterrupt, SystemExit):
#     stream.stop_stream()
#     stream.close()
#     timeout = time.time()
#     run = False
        
# stream.stop_stream()
# stream.close()

# # import pyaudio
# # import wave

# # chunk = 1024  # Record in chunks of 1024 samples
# # sample_format = pyaudio.paInt16  # 16 bits per sample
# # channels = 2
# # fs = 44100  # Record at 44100 samples per second
# # seconds = 10
# # filename = "output.wav"

# # p = pyaudio.PyAudio()  # Create an interface to PortAudio

# # print('Recording')

# # stream = p.open(format=sample_format,
# #                 channels=channels,
# #                 rate=fs,
# #                 frames_per_buffer=chunk,
# #                 input=True)

# # frames = []  # Initialize array to store frames

# # # Store data in chunks for 3 seconds
# # for i in range(0, int(fs / chunk * seconds)):
# #     data = stream.read(chunk)
# #     frames.append(data)

# # # Stop and close the stream 
# # stream.stop_stream()
# # stream.close()
# # # Terminate the PortAudio interface
# # p.terminate()

# # print('Finished recording')

# # # Save the recorded data as a WAV file
# # wf = wave.open(filename, 'wb')
# # wf.setnchannels(channels)
# # wf.setsampwidth(p.get_sample_size(sample_format))
# # wf.setframerate(fs)
# # wf.writeframes(b''.join(frames))
# # wf.close()

# from Net_Architectures.tr_model import create_model

# model = create_model()
# model.summary()
# from keras.models import load_model
# model = load_model('./Models/tr_model.h5')
# from Utils.audio import record_audio
# record_audio(1.3, '6.wav')
# import numpy as np
# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# a[9:9+5] = 1
# print(a)

# import matplotlib.pyplot as plt
# import numpy as np
# from pydub import AudioSegment
# d_a = AudioSegment.from_wav('raw_data/backgrounds/1.wav')
# d_a = d_a - 20
# d = d_a.get_array_of_samples()
# print(np.array(d).shape)
# plt.plot(d)
# plt.show()

from Utils.train_utils import TrainingExamplesGenerator

G = TrainingExamplesGenerator()
G.load_data(path='raw_data')
G.generate_examples(1,saved=True)
# print(len(G.backgrounds))
# print(len(G.positives))
# print(len(G.negatives))
# from Utils.td_utils import get_wav_info
# from pydub import AudioSegment
# d1 = AudioSegment.from_wav