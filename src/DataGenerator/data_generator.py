import os
import numpy as np
from pydub import AudioSegment
from tensorflow.keras.utils import Sequence

from Utils.td_utils import load_raw_audio, match_target_amplitude, graph_spectrogram

class SpeechDataGenerator(Sequence):
    """
    Use this to generate training examples
    """

    data_point_per_milisecond = 44.100

    def __init__(self, seed_list=[0], reseed_at_each_batch=False, batch_size=10, 
            positive_source='', negative_source='', background_source='',
            random_seed=0, target_db=None, audio_length_ms=10000,
            output_sequence_length=1375, active_length=50):
        """
        seed_list: list of int for seeding, seeded at the start of each batch
        """
        self.batch_size = batch_size
        self.seed_list=[]
        self.reseed_at_each_batch=reseed_at_each_batch
        self.positive_source = positive_source
        self.negative_source = negative_source
        self.background_source = background_source
        self.target_db = target_db
        self.active_length = active_length
        self.audio_length = audio_length_ms
        self.Ty = output_sequence_length
        self.Tx = int(audio_length_ms * self.data_point_per_milisecond)
        self.__positive_files, self.__negative_files, self.__background_files = self.__load_audio_files()
        self.__previous_segments = []
        self.on_epoch_end()
        

    def __len__(self):
        return len(self.seed_list)

    def __getitem__(self, index):
        X = []
        Y = []
        if self.reseed_at_each_batch:
            np.random.seed(self.seed_list[index])
        for i in range(1,self.batch_size):
            x, y = self.__create_training_example()
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)


    def on_epoch_end(self):
        np.random.shuffle(self.seed_list)

    def __load_audio_files(self):
        """
        load the audio file names from the folder
            
        return: array of activates, negatives, backgrounds
        """
        activates = [x for x in os.listdir(self.positive_source) if x.endswith('wav')]
        backgrounds = [x for x in os.listdir(self.background_source) if x.endswith('wav')]
        negatives = [x for x in os.listdir(self.negative_source) if x.endswith('wav')]
        return activates, negatives, backgrounds

    def __load_audio(self, path):
        """
        load audio file from path
        params: path=> file path
                db  => target decibel level
        return: audioSegment audio
        """
        audio = AudioSegment.from_wav(path)
        if self.target_db is not None:
            audio = match_target_amplitude(audio, self.target_db)
        return audio
    
    def __get_random_time_segment(self, segment_ms):
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
        
        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
        
        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """
        
        segment_start = np.random.randint(low=0, high=self.audio_length-segment_ms)   # Make sure segment doesn't run past the 10sec background 
        segment_end = segment_start + segment_ms - 1
        return (segment_start, segment_end)
    
    def __is_overlapping(self, segment_time):
        """
        Checks if the time of a segment overlaps with the times of existing segments.
        
        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        
        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """
        
        segment_start, segment_end = segment_time
        
        # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
        overlap = False
        
        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for previous_start, previous_end in self.__previous_segments:
            if previous_start <= segment_start <= previous_end or previous_start <= segment_end <= previous_end:
                overlap = True

        return overlap

    def __get_y_labels(self, y, segment_end_ms):
        """
        Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
        should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
        50 following labels should be ones.
        
        
        Arguments:
        segment_end_ms -- the end time of the segment in ms
        
        Returns:
        y -- updated labels
        """
        
        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * self.Ty / self.audio_length)
        
        # Add 1 to the correct index in the background label (y)
        start = segment_end_y + 1
        end = segment_end_y + self.active_length + 1
        y[start:end,:] = 1
        
        return y

    def __insert_audio_clip(self, background, audio_clip):
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the 
        audio segment does not overlap with existing segments.
        
        Arguments:
        background -- a 10 second background audio recording.  
        audio_clip -- the audio clip to be inserted/overlaid. 
        
        Returns:
        new_background -- the updated background audio
        """
        
        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)
        
        # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
        # the new audio clip. (≈ 1 line)
        segment_time = self.__get_random_time_segment(segment_ms)
        
        # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
        # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
        while self.__is_overlapping(segment_time):
            segment_time = self.__get_random_time_segment(segment_ms)

        # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
        self._previous_segments.append(segment_time)
        
        # Step 4: Superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])
        
        return new_background, segment_time

    def __pad_sequence(self, sequence):
        """
        Make sure the sequence has correct length by padding with zeros at the end or cutting it 
        Args - sequence: sequence to be padded or cutted
        """
        length = len(sequence)
        if length == self.Tx:
            return np.array(sequence)
        elif length < self.Tx:
            sequence.extend([0 for i in range(self.Tx-length)])
            return np.array(sequence)
        else:
            return np.array(sequence[:self.Tx])

    def __create_training_example(self, saved=False, name='untitled'):
        """
        Creates a training example with a given background, activates, and negatives.
        
        Arguments:
        name -- file name to be saved
        
        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """

        # choose background
        random_index = np.random.randint(len(self.__background_files))
        background = self.__load_audio(self.background_source + '/' + self.__background_files[random_index])

        # initialize y
        y = np.zeros((self.Ty,1))
        
        # Make background quieter
        background = background - 20

        # Step 2: Initialize segment times as an empty list (≈ 1 line)
        self._previous_segments = []
        
        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = np.random.randint(1, 5)
        random_indices = np.random.randint(len(self.__positive_files), size=number_of_activates)
        random_activates = [self.__load_audio(self.positive_source + '/' + self.__positive_files[i]) for i in random_indices]
        
        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for random_activate in random_activates:
            # Insert the audio clip on the background
            background, segment_time = self.__insert_audio_clip(background, random_activate)
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time
            # Insert labels in "y"
            y = self.__get_y_labels(y, segment_end)

        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = np.random.randint(1, 3)
        random_indices = np.random.randint(len(self.__negative_files), size=number_of_negatives)
        random_negatives = [self.__load_audio(self.negative_source + '/' + self.__negative_files[i]) for i in random_indices]

        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background 
            background, _ = self.__insert_audio_clip(background, random_negative)
        
        # Standardize the volume of the audio clip 
        background = match_target_amplitude(background, -20.0)
        
        # Export new training example 
        if saved:
            file_handle = background.export(name + ".wav", format="wav")
            print("File ({}.wav) was saved in your directory.".format(name))
        
        # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
        sequence = self.__pad_sequence(background.get_array_of_samples())
        x = graph_spectrogram(sequence).swapaxes(0,1)
        
        return x, y
    
    def create_audio(self, name):
        self.__create_training_example(saved=True,name=name)