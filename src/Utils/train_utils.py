import numpy as np
import os

from pydub import AudioSegment
from Utils.td_utils import load_raw_audio, match_target_amplitude, graph_spectrogram

class TrainingExamplesGenerator:
    """
    Use this to generate training examples
    """

    data_point_per_milisecond = 44.100

    def __init__(self, background_audio_length_ms=10000, no_of_ones_sequence=50, output_sequence_length=1375, seed=0, log=False):
        self.positives = []
        self.negatives = []
        self.backgrounds = []
        self.audio_length = background_audio_length_ms
        self.Tx = int(background_audio_length_ms * self.data_point_per_milisecond)
        self._prev_segments = []
        self.no_of_ones = no_of_ones_sequence
        self.Ty = output_sequence_length
        self.log = log
        self.seed = seed
        # Set the random seed
        np.random.seed(0)
        
    def load_data(self, path = '', target_db=None):
        """
        load the data from the path. The path has positives, negatives, and backgrounds folders 
        each containing respective audio files
        Args: 
            path: string => folder path
            log: flag => whether to return the dataset information
            target_db: int => target decible level of the loaded audio
        """
        self.positives, self.negatives, self.backgrounds = self.__load_raw_audio(path,target_db)
        if self.log:
            print("background length: " + str(len(self.backgrounds[0])),"\n")
            print("Number of background: " + str(len(self.backgrounds)),"\n")
            print("Number of activate examples: " + str(len(self.positives)),"\n")
            print("Number of negative examples: " + str(len(self.negatives)),"\n")

    @staticmethod
    def __load_raw_audio(path, db):
        """
        load the audio datas from the folder
        Args: path -- data path
              db -- target decibel level
            
        return: array of activates, negatives, backgrounds
        """
        activates = []
        backgrounds = []
        negatives = []
        for filename in os.listdir(path + "/positives"):
            if filename.endswith("wav"):
                activate = AudioSegment.from_wav(path + "/positives/"+filename)
                if db is not None:
                    activate = match_target_amplitude(activate, db)
                activates.append(activate)
        for filename in os.listdir(path + "/backgrounds"):
            if filename.endswith("wav"):
                background = AudioSegment.from_wav(path + "/backgrounds/"+filename)
                if db is not None:
                    background = match_target_amplitude(background, db)
                backgrounds.append(background)
        for filename in os.listdir(path + "/negatives"):
            if filename.endswith("wav"):
                negative = AudioSegment.from_wav(path + "/negatives/"+filename)
                if db is not None:
                    negative = match_target_amplitude(negative, db)
                negatives.append(negative)
        return activates, negatives, backgrounds

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
        for previous_start, previous_end in self._previous_segments:
            if previous_start <= segment_start <= previous_end or previous_start <= segment_end <= previous_end:
                overlap = True

        return overlap

    def insert_audio_clip(self, background, audio_clip):
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
        end = segment_end_y + self.no_of_ones + 1
        y[start:end,:] = 1
        
        return y

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
        random_index = np.random.randint(len(self.backgrounds))
        background = self.backgrounds[random_index]

        # initialize y
        y = np.zeros((self.Ty,1))
        
        # Make background quieter
        background = background - 20

        # Step 2: Initialize segment times as an empty list (≈ 1 line)
        self._previous_segments = []
        
        # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
        number_of_activates = np.random.randint(1, 5)
        random_indices = np.random.randint(len(self.positives), size=number_of_activates)
        random_activates = [self.positives[i] for i in random_indices]
        
        # Step 3: Loop over randomly selected "activate" clips and insert in background
        for random_activate in random_activates:
            # Insert the audio clip on the background
            background, segment_time = self.insert_audio_clip(background, random_activate)
            # Retrieve segment_start and segment_end from segment_time
            segment_start, segment_end = segment_time
            # Insert labels in "y"
            y = self.__get_y_labels(y, segment_end)

        # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
        number_of_negatives = np.random.randint(1, 3)
        random_indices = np.random.randint(len(self.negatives), size=number_of_negatives)
        random_negatives = [self.negatives[i] for i in random_indices]

        # Step 4: Loop over randomly selected negative clips and insert in background
        for random_negative in random_negatives:
            # Insert the audio clip on the background 
            background, _ = self.insert_audio_clip(background, random_negative)
        
        # Standardize the volume of the audio clip 
        background = match_target_amplitude(background, -20.0)
        if self.log:
            print('{}\t{}\t{}'.format(number_of_activates, number_of_negatives, number_of_activates+number_of_negatives), end='')
        # Export new training example 
        if saved:
            file_handle = background.export(name + ".wav", format="wav")
            print("File ({}.wav) was saved in your directory.".format(name))
        
        # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
        sequence = self.__pad_sequence(background.get_array_of_samples())
        x = graph_spectrogram(sequence).swapaxes(0,1)
        
        return x, y

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

    def create_an_example(self,name='',seed=0):
        """
        create the training example and save it
        Arg: name: str => path of the file to be saved
        """
        np.random.seed(seed)
        self.__create_training_example(name=name, saved=True)

    def generate_examples(self, count, batch_size=10, saved=False, path=''):
        """
        Creates a training example with a given background, activates, and negatives.
        
        Arguments:
        count -- number of examples to generate
        saved -- save the audio files as wav or not
        path -- path to save the files
        
        Returns:
        x -- the spectrogram of the training examples
        y -- the label at each time step of the spectrograms
        """

        X = []
        Y = []
        if self.log:
            print('count\tpos\tneg\ttot\n')
        while True:
            np.random.seed(self.seed)
            for i in range(1,count+1):
                if self.log: 
                    print('{}\t'.format(i), end='')
                x, y = self.__create_training_example(name=path+str(i), saved=saved)
                if self.log: 
                    print('')
                X.append(x)
                Y.append(y)
                if i%batch_size == 0 and i != 0:
                    yield np.array(X), np.array(Y)
                X = []
                Y = []
            print('complete data generation...')
        