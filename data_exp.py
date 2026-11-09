import wfdb

signal = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/100')
print(signal.p_signal.shape)
print(signal.fs)
print(signal.sig_name)
print('')

wfdb.plot_wfdb(record=signal, title='MIT-BIH Record 100')