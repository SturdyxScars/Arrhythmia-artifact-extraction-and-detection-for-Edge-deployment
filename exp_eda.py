import wfdb, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import welch

rec = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/203')
ann = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/203', 'atr')
fs = rec.fs
x = rec.p_signal[:, 0]

print(Counter(ann.symbol))                      # class imbalance

rr = np.diff(ann.sample) / fs                   # RR intervals (s)
f, p = welch(x, fs, nperseg=4096)               # where is the noise?

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(rr, bins=100); ax[0].set_title('RR intervals')
ax[1].semilogy(f, p); ax[1].set_title('PSD'); ax[1].set_xlim(0, 100)
ax[2].hist(x, bins=100); ax[2].set_title('Amplitude')
plt.show()