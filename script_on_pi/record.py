import os
import numpy as np
import pickle
from scipy import signal
from scipy.signal import resample
import pyaudio
import sounddevice as sd
import soundfile as sf

# Function to convert Hz to Mels
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

# Function to convert Mels to Hz
def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def create_filter_bank(mels, sampling_rate, n_fft, max_freq):
    '''
        Create filter banks for mel spectrogram.

    '''

    # Number of mels
    n_filters = mels

    # Calculate Mel points
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(max_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert Hz to FFT bins
    bin_points = np.floor((n_fft + 1) * hz_points / sampling_rate).astype(int)

    # Create filter bank
    filter_bank = np.zeros((n_filters, int(np.floor(n_fft / 2 + 1))))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            filter_bank[i - 1, j] = (j - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        for j in range(center, right):
            filter_bank[i - 1, j] = (bin_points[i + 1] - j) / (bin_points[i + 1] - bin_points[i])

    return filter_bank
    
# Function to find a USB device mic
def find_usb_device():
    input_devices = sd.query_devices()
    usb_devices = [device for device in input_devices if 'USB' in device['name']]
    
    if usb_devices:
        return usb_devices[0]['name']
    else:
        print("No USB mic device found!")
        return None


# Function to record audio 
def record(channels, duration, new_sample_rate, input_sample_rate, device=None):

    if device is not None:
        print(f"Recording from {device}...")
        p = pyaudio.PyAudio()

        # Find the input device index using the provided device name
        input_device_index = None
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i)["name"] == device:
                input_device_index = i
                break

        if input_device_index is not None:
            stream = p.open(format=pyaudio.paFloat32,
                            channels=channels,
                            rate=input_sample_rate,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=int(duration * input_sample_rate))
            audio_data = np.frombuffer(stream.read(int(duration * input_sample_rate)), dtype=np.float32)
            stream.stop_stream()
            stream.close()

            # Perform real-time resampling using scipy
            print ("Resampling...")
            resampled_audio = resample(audio_data, int(duration * new_sample_rate))
            print ("Done resampling.\n")

            return resampled_audio
        else:
            print(f"Device {device} is not found.")
            p.terminate()
            return None
    else:
        print("Please insert a USB device.")
        return None
    
# Save the audio as a waveform
def save_audio_data(filename, audio_data, sample_rate):
    sf.write(filename, audio_data, sample_rate)

# Save data as a pickle file
def save_data_to_pickle(filename, data_files):
    with open(filename, 'wb') as file:
        pickle.dump(data_files, file)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
