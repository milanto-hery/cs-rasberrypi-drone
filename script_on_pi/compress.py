import os
import numpy as np
import pickle
        
def compress2d(spectrogram, R, seed):
    """
    Function to compress the spectrogram of the audio by randomly sampling the original original.
    - spectrogram: 2d siganl version of the audio signal
    - R: The percentage value from which we would like to sample from the original
    """ 
    # Get the shape of the original spectrogram 
    ny,nx=spectrogram.shape
    
    #Get the number of sample
    m = round(nx * ny * R) # e.g: R=0.25 => 25% of samples
    np.random.seed(seed)
    ri_vector = np.random.choice(nx * ny, m, replace=False) # random sample of indices
    y = spectrogram.T.flat[ri_vector]
    
    return y, seed, nx, ny, m
      
def compress1d(audio_data, R, seed):
    """
    Function to compress the audio by randomly sampling from a specific percentage of the original signal.
    audio_data: audio signal (in wav format) to be compressed
    R: The percentage value from which we would like to sample from the original
    """
    #Find the number of samples in the original data 
    n = len(audio_data)
    np.random.seed(seed)

    # Extract small sample of the signal
    m = int(n*R)
    np.random.seed(seed)

    # Perform the sensing matrix
    ri = np.random.choice(n, m, replace=False) # random sample of indices

    # Generate compressed measurements
    y = audio_data[ri]

    return y, seed, n, m
