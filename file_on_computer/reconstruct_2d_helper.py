import sys
import os
import pickle
import numpy as np
import scipy.fftpack as fft
import cvxpy as cvx
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time

# Define Discrete Cosie Transform for 2d signal from 1d dct
def dct2(x):
    return fft.dct(fft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return fft.idct(fft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# Function to reconstruct signal 2D 
def reconstruct_2d(y, seed, nx, ny, m, solver):
    np.random.seed(seed)
    timer_s = time.perf_counter()
    # Generate sensing matrix
    ri_vector = np.random.choice(nx * ny, m, replace=False)
    A = np.kron(
        fft.idct(np.identity(nx), norm='ortho', axis=0),
        fft.idct(np.identity(ny), norm='ortho', axis=0)
        )
    A = A[ri_vector,:] # 
    timer_e = time.perf_counter()
    m_times = timer_e - timer_s
    if solver == 'lasso':
        timer_s1 = time.perf_counter()
        prob = Lasso(alpha=1e-5)
        prob.fit(A, y) # solve y=Ax
        x_lasso = idct2(prob.coef_.reshape(nx, ny)).T
        x = np.reshape(x_lasso, (ny, nx))
        timer_e1 = time.perf_counter()
        time_lasso = timer_e1-timer_s1
        time_ex1 = time_lasso+m_times
        print(f"Total time of reconstruction using {solver}: {time_ex1} seconds")  
        return x
        
    elif solver == 'cvx':
        timer_s2 = time.perf_counter()   
        vx = cvx.Variable(nx*ny)
        objective = cvx.Minimize(cvx.norm(vx))
        constraint = [A*vx == y]
        prob = cvx.Problem(objective, constraint)
        res = prob.solve(verbose=False, solver='ECOS')
        beta = np.array(vx.value).squeeze()
        x1 = idct2(beta.reshape((nx, ny)).T)
        x = np.reshape(x1, (ny, nx))
        timer_e2 = time.perf_counter()  
        time_cvx = timer_e2-timer_s2
        time_ex2 = time_cvx+m_times
        print(f"Total time of reconstruction using {solver}: {time_ex2} seconds") 
        return x
        
    elif solver == 'omp':
        timer_s3 = time.perf_counter()                                                    
        prob = OrthogonalMatchingPursuit()
        prob.fit(A, y)
        s = idct2(prob.coef_.reshape((nx, ny)).T)
        x = np.reshape(s, (ny, nx))
        timer_e3 = time.perf_counter() 
        time_omp = timer_e3-timer_s3
        time_ex3 = time_omp+m_times
        print(f"Total time of reconstruction using {solver}: {time_ex3} seconds")
  
        return x
    else:
        raise ValueError("Please specify solver!!!.")
    

# Function to load the data from pickle file
def load_pickle_file(folder_name, file_name):
    file_path = os.path.join(folder_name, file_name)

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    return loaded_data

# Function to load waveform
def load_wav_from_folder(folder_path, filename):
    file_path = os.path.join(folder_path, filename)

    if os.path.exists(file_path):
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        return audio_data, sample_rate
    else:
        print(f"File {filename} not found in folder {folder_path}")
        return None, None

# Function to extract all elements required for reconstruction   
def get_compressed_data_2d(folder_name, file_name):

    file_path = os.path.join(folder_name, file_name)
    # Load the compressed data from the pickle file
    with open(file_path, 'rb') as file:
        compressed_data = pickle.load(file)

    # Extract each element from the pickle data
    y_2d = compressed_data['y_2d']
    seed_2d = compressed_data['seed_2d']
    nx_2d = compressed_data['nx_2d']
    ny_2d = compressed_data['ny_2d']
    m_2d = compressed_data['m_2d']

    # Display extracted elements y, seed, n and m
    print("Compressed measurement y :", y_2d.shape)
    print("Value of the random seed :", seed_2d)
    print("Width of the spectrogram :", nx_2d)
    print("Height of the spectrogram:", ny_2d)
    print("Number of small samples  :", m_2d)
    return y_2d, seed_2d, nx_2d, ny_2d, m_2d
    
# Plot comparison (original vs reconstructed)   
def plot_comparison(original, reconstructed, sr, R):
    fig, axes = plt.subplots(1, 3, figsize=(14, 2))

    # Plot Original spectrogram
    librosa.display.specshow(original, y_axis='mel', x_axis='time',sr=sr, ax=axes[0])
    axes[0].set_title("Original spectrogram")

    # Plot the random samples by masking the original
    mask = np.zeros(original.shape)
    ri = np.random.choice(original.shape[0] * original.shape[1], int((original.shape[0] * original.shape[1])*R) , replace=False)
    mask.T.flat[ri] = 255
    Xmask = 255 * np.ones(original.shape)
    Xmask.T.flat[ri] = original.T.flat[ri]
    librosa.display.specshow(Xmask, y_axis='mel', x_axis='time',sr=sr, ax=axes[1])
    axes[1].set_title(f'{R*100}% random samples')

    # Plot reconstructed spectrogram
    librosa.display.specshow(reconstructed, y_axis='mel', x_axis='time',sr =sr, ax=axes[2])
    axes[2].set_title('Compressed spectrogram')
    plt.show()
    
def run_reconstruction_2d(original, folder_name, file_name, sr, R, solvers):
    reconstructed_outputs = []  # empty list to store reconstructed spectrograms from differents solver
    for i, solver in enumerate(solvers):
        y, seed, nx, ny, m = get_compressed_data_2d(folder_name, file_name)
        # Compress the original spectrogram
        reconstructed = reconstruct_2d(y, seed, nx, ny,m, solver)
        reconstructed_outputs.append(reconstructed)
        # Calculate MSE
        mse = mean_squared_error(original, reconstructed)
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        plot_comparison(original, reconstructed, sr, R)

    return reconstructed_outputs
