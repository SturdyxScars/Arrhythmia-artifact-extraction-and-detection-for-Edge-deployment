import numpy as np
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
import wfdb

def dsp_processing(record_path):
    # Load record
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    # Use only the first lead (MLII usually)
    raw_signal = record.p_signal[:, 0]
    lead_name = record.sig_name[0]
    
    # 1. Bandpass Filter (0.5 - 40 Hz)
    # 0.5Hz removes baseline wander (breathing)
    # 40Hz removes powerline interference and high frequency noise
    lowcut = 0.5
    highcut = 40.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(3, [low, high], btype='band')

    # sos form butterworth filter
    sos = scipy_signal.tf2sos(b, a)
    print(sos.shape)
    filtered_signal = scipy_signal.filtfilt(b, a, raw_signal)
    print(filtered_signal.shape)
    sos_filtered = scipy_signal.sosfilt(sos, raw_signal)
    
    # 2. Derivative Filter (to highlight QRS complexes)
    # Discrete derivative: y[n] = x[n] - x[n-1]
    derivative = np.diff(filtered_signal)
    
    # 3. Squaring Function (to further highlight QRS and make all positive)
    squared = derivative**2
    
    # 4. Moving Average Filter (Integration)
    window_size = int(0.12 * fs) # 120ms window
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Visualization
    n_samples = 1500 # ~4 seconds
    time = np.arange(n_samples) / fs
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(5, 1, 1)
    plt.plot(time, raw_signal[:n_samples])
    plt.title(f'Raw {lead_name} Signal')
    plt.grid(True)
    
    plt.subplot(5, 1, 2)
    plt.plot(time, filtered_signal[:n_samples])
    plt.title('Bandpass Filtered (0.5-40 Hz)')
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.plot(time, sos_filtered[:n_samples])
    plt.title("Bandpass Filtered (0.5-40 Hz) sos transfer function")
    plt.grid(True)
    
    plt.subplot(5, 1, 4)
    plt.plot(time, squared[:n_samples])
    plt.title('Squared Derivative (Highlighting QRS)')
    plt.grid(True)
    
    plt.subplot(5, 1, 5)
    plt.plot(time, integrated[:n_samples])
    plt.title('Moving Average Integration')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dsp_results.png')
    print("DSP plot saved as dsp_results.png")

if __name__ == "__main__":
    dsp_processing('mit-bih-arrhythmia-database-1.0.0/100')
