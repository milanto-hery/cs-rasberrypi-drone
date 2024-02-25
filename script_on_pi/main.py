
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from os.path import basename
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from record import *
from compress import *
from drive import *
import zipfile 
from zipfile import ZipFile
import time

def main():
    # Authenticate with Google Drive
    drive_folder_id = 'Your drive folder ID' # your drive folde ID
    gauth = GoogleAuth()
    gauth.CommandLineAuth()  # Use CommandLineAuth()
    drive = GoogleDrive(gauth)
    # Ensure the specified folder exists
    time_str = datetime.now().strftime("%H%M%S")
    num_files = 1 # specify the number of files you want to record
    channels=1
    duration = 4 # set the duration of recording
    R = 0.15 # compression ratio
    sample_rate=44100 # The device is only record at the default sampling rate 44100
    new_sample_rate = 22050 #Put new sample rate if resampling is required
    seed=42
    drive_folder_name = f'Gibbon_{R*100}_{new_sample_rate}' # folder to save files in the drive name as SpeciesID_compressionRatio_SampleRate
    pi_day_folder=os.path.join("Recording",drive_folder_name) #Let's rename the  local folder in the Pi the same as in the drive folder
    # Create a local folder in the Pi      
    if not os.path.exists(pi_day_folder):
        os.makedirs(pi_day_folder)

    # Find a USB mic
    microphone_name = find_usb_device()
    print(f'[INFO] USB device detected.\n')
    if microphone_name is None:
        return

    print('[INFO] Creating filter bank')
    f_bank = create_filter_bank(128,new_sample_rate,1024,new_sample_rate/2)
    print('Filter bank shape: ',f_bank.shape)
    print('done.\n')

    for i in range(num_files):

        # Record audio
        rec_s = datetime.now().strftime("%H%M%S")
        print(f'Recording and resampling {i+1} starts... duration: {duration}s, sr: {new_sample_rate} Hz')
        print(f'Start time: {rec_s}')
        
        # Resampling
        resampled_audio= record(channels, duration, new_sample_rate, sample_rate, sd.query_devices(microphone_name)['name'])
        rec_e = datetime.now().strftime("%H%M%S")
        print(f'End time: {rec_e}')

        print("\nAudio duration check:", len(resampled_audio)/new_sample_rate  )

        # Create mel spectrogram
        print('\nCreating power spectrogram...')
        f, t, Sxx = signal.spectrogram(resampled_audio, new_sample_rate, nperseg = 1024,noverlap=512, nfft=1024)

        print("Done.")
        print('Spectrogram shape:', Sxx.shape) 

        print('\nMultiplying filter bank and spectrogram...')
        mel_scale_spec = np.matmul(f_bank,Sxx)
        print('Mel spectrogram shape: ',mel_scale_spec.shape)
        print("done.\n")

        print('Creating log scale mel spectrogram')
        # POWER TO DB CONVERSION
        # Librosa replacement
        # -------------------------------------------------------------------------
        S = np.asarray(mel_scale_spec)
        amin: float = 1e-10
        ref = 1.0
        top_db = 80.0

        if np.issubdtype(S.dtype, np.complexfloating):
            magnitude = np.abs(S)
        else:
            magnitude = S

        if callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(magnitude)
        else:
            ref_value = np.abs(ref)

        Y_log_scale: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
        Y_log_scale -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            Y_log_scale = np.maximum(Y_log_scale, Y_log_scale.max() - top_db)
            scaler = StandardScaler()
            Y_log_scale = scaler.fit_transform(Y_log_scale)
        # -------------------------------------------------------------------------

        print("Y_log_scale.shape", Y_log_scale.shape)
        print("done.\n")

        # Generate file names based on the current time
        original_filename = os.path.join(pi_day_folder, f"{i+1}_ORIG_{time_str}"+".wav")
        compressed_1D_filename = os.path.join(pi_day_folder, f"{i+1}_COMP_{time_str}"+".pkl")
        spectrogram_2D_filename = os.path.join(pi_day_folder, f"{i+1}_SPEC_{time_str}"+".pkl")
        spectrogram_2D_raw_filename = os.path.join(pi_day_folder, f"{i+1}_RawSpec_{time_str}"+".pkl")
        
        # Compress and save all the data required
        comp_s = datetime.now().strftime("%H%M%S")
        print(f'Compression start at {comp_s}')
        print(f'R samples: {R*100}')
        # Compress 2D
        y_2d, seed_2d, nx_2d, ny_2d, m_2d = compress2d(Y_log_scale, R, seed)

        # Compress 1D
        y_1d, seed_1d, n_1d, m_1d = compress1d(resampled_audio, R, seed)
        comp_e = datetime.now().strftime("%H%M%S")
        print(f'Compression finished at {comp_e}')

        # Save the audio recording
        print('Saving original data...')
        save_audio_data(original_filename, resampled_audio, new_sample_rate)

        # Save the compressed audio
        print('Saving compressed audio...')
        save_data_to_pickle(compressed_1D_filename, {'y_1d': y_1d, 'seed_1d':seed_1d, 
            'n_1d': n_1d,'m_1d':m_1d})

        # Save the compressed spectrogram
        print('Saving compressed spectrogram...')
        save_data_to_pickle(spectrogram_2D_filename, {'y_2d': y_2d, 'seed_2d':seed_2d, 
            'nx_2d': nx_2d, 'ny_2d':ny_2d, 'm_2d':m_2d})


        print('Saving raw log spectrogram...')
        save_data_to_pickle(spectrogram_2D_raw_filename, {'Y_log_scale': Y_log_scale})

        zip_original_filename = os.path.join(pi_day_folder, f"{i+1}_zip_ORIG_{time_str}"+".zip")
        zip_compressed_1D_filename = os.path.join(pi_day_folder, f"{i+1}_zip_COMP_{time_str}"+".zip")
        zip_compressed_2D_filename = os.path.join(pi_day_folder, f"{i+1}_zip_SPEC_{time_str}"+".zip")
        zip_original_Raw_filename = os.path.join(pi_day_folder, f"{i+1}_zip_RawSpec_{time_str}"+".zip")
        
        #zip_original
        # -----------------------------------------------------------------------
        zip_location = zip_original_filename
        text_file = os.path.join(pi_day_folder, f"{i+1}_ORIG_{time_str}"+".wav")
         
        with zipfile.ZipFile(zip_location, 'w') as zipObj:
            zipObj.write(text_file, basename(text_file))

        #zip_compressed_1D
        # -----------------------------------------------------------------------
        zip_location = zip_compressed_1D_filename
        text_file = os.path.join(pi_day_folder, f"{i+1}_COMP_{time_str}"+".pkl") 
         
        with zipfile.ZipFile(zip_location, 'w') as zipObj:
            zipObj.write(text_file, basename(text_file))

        #zip_compressed_Raw_spectrogram
        # -----------------------------------------------------------------------
        zip_location = zip_original_Raw_filename
        text_file = os.path.join(pi_day_folder, f"{i+1}_RawSpec_{time_str}"+".pkl") 
         
        with zipfile.ZipFile(zip_location, 'w') as zipObj:
            zipObj.write(text_file, basename(text_file))

        #zip_compressed_2D
        # -----------------------------------------------------------------------
        zip_location = zip_compressed_2D_filename
        text_file = os.path.join(pi_day_folder, f"{i+1}_SPEC_{time_str}"+".pkl") 
         
        with zipfile.ZipFile(zip_location, 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipObj:
            zipObj.write(text_file, basename(text_file))
        
        print('done.\n')
        # Wait for 1 seconds before uploading to drive
        time.sleep(1)
        #upload to google drive
        # ----------------------------------------------------------------------
        print('[INFO] Uploading into google drive')
        save_files_to_drive(drive, pi_day_folder, drive_folder_name, drive_folder_id)
        print("done.\n")         
        print(f'Recording-Compression-Uploading file {i+1} sucessful!')
        print('====================================================')
        # Wait for 1 second before starting the next recording
        time.sleep(1)

if __name__ == "__main__":
    main()
