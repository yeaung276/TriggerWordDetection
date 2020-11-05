import pyaudio
import wave
import numpy as np

def get_audio_input_stream(callback, input_device_index=None, fs=44100, chunk_samples=22050):
    """
    create pyaudio object to stream the audio data (nonblocking mode)
    
    Agrs: 
        callback: callback function when the chunk_samples has been recorded
        input_device_index: index of audio input device available
        fs: sampling frequency
        chunk_samples: number of data to be collected before callback 

    return:
        PyAudio object

    """
    stream = pyaudio.PyAudio().open(
        format = pyaudio.paInt16,
        channels = 1,
        rate = fs,
        input = True,
        frames_per_buffer = chunk_samples,
        input_device_index = input_device_index,
        stream_callback = callback)
    return stream


def show_devices_info():
    """
    show the availabel audio input devices
    """
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
      dev = p.get_device_info_by_index(i)
      print((i,dev['name'],dev['maxInputChannels']))

def record_audio(seconds, output_file, **args):
    """
    record the audio from the default audio input device
    Args:
        seconds: the number of seconds to record the audio
        output_file: file name to save the recorded file
        chunk: frame_per_buffer input to the pyaudio
        channels: no of channel to record
        fs: sampling rate

    return: 
        none
    """
    chunk = args.get('chunk', 1024)  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = args.get('channels', 1)
    fs = args.get('fs', 44100)  # Record at 44100 samples per secons

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


