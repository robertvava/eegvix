import numpy as np

def compute_fft(eeg_data):
    """
    Compute the absolute value of the Fast Fourier Transform 
    :return: Numpy array containing the absolute values of the transformed array. 
    """

    fft_data = np.abs(np.fft.rfft(eeg_data, axis=-1))  
    return fft_data