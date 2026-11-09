# Learning ECG EDA, DSP, and realtime arrhythmia detection

This guide uses the local MIT-BIH Arrhythmia Database and the starter files `data_exp.py`, `eda_detailed.py`, `dsp_processing.py`, and `realtime_detection.py`. It is a study and engineering workflow, not a clinical device specification. A detector built from this guide is a research prototype until it has been independently validated and developed under the applicable medical-device quality and risk processes.
-  A lead is not the electrode closest to the heart. An electrode is a sensor on the skin. A lead is an electrical view: a voltage measured or derived from electrodes, giving one direction from which to view the heart’s electrical activity. That’s why a 12-lead ECG uses 10 electrodes to produce 12 leads.
-  V1–V6 are the six standard chest (precordial) leads. Their positive electrodes occupy different positions across the chest, so each provides a different view. They are not defined by being closer to the heart. I, II, and III are the standard limb leads.
-  MLII is a modified Lead II-style electrical view recorded with electrodes placed on the torso. The PhysioNet notes describe it as often showing prominent normal QRS complexes, but that doesn’t mean it is physically closest to the heart or always best.
- Standard V5 is positioned on the left side of the chest. In MIT-BIH, a channel labelled V5 is a modified chest lead, so check the record documentation before assuming standard 12-lead placement.

## 1. What the current output means

Your code reports `(650000, 2)`, `360`, and `['MLII', 'V5']` for record 100. This means there are 650,000 simultaneous sample rows, two channels, and a sampling rate of 360 samples/second/channel. The recording duration is `650000 / 360 = 1805.56 s`, about 30.1 minutes. `MLII` is modified lead II; the second lead varies across records (record 100 uses V5). Do not assume that all recordings have the same second lead or that one lead is always best. MIT-BIH contains 48 approximately half-hour two-channel recordings from 47 people, sampled at 360 Hz; records 201 and 202 are from the same person. Some records were selected to include rare arrhythmias, so class frequencies in this database are not population prevalence. [PhysioNet database description](https://physionet.org/content/mitdb/1.0.0/) [record and lead notes](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)

For `100.hea`, the signal lines state 360 Hz, 650,000 samples, and each channel's calibration and units. Prefer `record.p_signal` for physical values (typically mV) and inspect `record.units`; do not assume raw integer codes are mV. A WFDB record is a matched group: `.hea` describes it, `.dat` stores samples, and `.atr` stores reference annotations. Load annotations with `wfdb.rdann(path, 'atr')`; annotation `sample` values are sample indices, not seconds. Convert with `sample / fs`.

## 2. Inspect before processing

Begin with metadata and integrity checks, then examine both long and short views. A few beats are not representative of a half-hour record.

```python
from pathlib import Path
from collections import Counter
import numpy as np
import wfdb

ROOT = Path('mit-bih-arrhythmia-database-1.0.0')
record_id = '100'
record = wfdb.rdrecord(str(ROOT / record_id))
ann = wfdb.rdann(str(ROOT / record_id), 'atr')
x, fs = record.p_signal, record.fs

print(record.sig_name, record.units, fs, x.shape, x.shape[0] / fs, 'seconds')
print('finite by lead:', np.isfinite(x).all(axis=0))
print('annotation symbols:', Counter(ann.symbol))
print('annotations:', len(ann.sample), 'rhythm/comments:', len(ann.aux_note))
for j, name in enumerate(record.sig_name):
    y = x[:, j]
    print(name, 'min/max/mean/std:', np.min(y), np.max(y), np.mean(y), np.std(y))
```

EDA sequence:

1. **Record inventory:** list every `.hea`, record ID, sample rate, signal count, signal names/units, sample count, duration, and missing or unreadable files. Confirm that annotations fall within the waveform. Treat records, not randomly sampled beats, as the natural split unit.
2. **Amplitude/integrity:** per lead compute min/max, mean, standard deviation, median, robust spread (`IQR` or `1.4826 × MAD`), NaN/Inf count, flat-line runs, repeated values, and near-rail samples. Inspect outliers in the waveform before deleting anything; ectopic beats and artifacts are not automatically bad data.
3. **Visual scales:** plot 5–10 seconds to inspect morphology; 1–5 minutes to see rhythm and drift; and downsampled full-record views for gaps, transitions, and long artifact episodes. Keep a raw trace visible beside any processed trace.
4. **Annotation-aware inspection:** overlay beat symbols on short strips and separately mark rhythm (`+` with `aux_note`) and signal-quality (`~`, subtype) events. Do not plot every annotation label over a 30-minute trace. Zoom around unusual classes and noisy intervals.
5. **Frequency and time-frequency views:** use Welch PSD for stationary summaries, then spectrograms for transient/intermittent interference. Welch PSD is `|FFT(window × segment)|²` averaged across overlapping segments; it answers “where is average power?” but does not identify when a burst happened. Convert axes and units correctly (`Hz`, and signal-units²/Hz).
6. **Compare channels and records:** compute pairwise correlation only as a descriptive measure; a low correlation can be due to lead geometry, not corruption. Inspect each lead's morphology and signal quality. MIT-BIH lead configurations vary and some records have reversed channels or tape-related timing skew; naive cross-lead fusion can therefore be wrong. [MIT-BIH directory notes](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)

Useful numerical summaries per fixed window (e.g., 2 s with 50% overlap): RMS, peak-to-peak range, robust baseline, derivative RMS, clipping/flatline fraction, and band powers. For a signal `x[n]`, RMS is `sqrt(mean(x[n]^2))`; first difference is `d[n]=x[n]-x[n-1]`. A large derivative can indicate a QRS or a motion spike; it is not an artifact label by itself.

## 3. Understand the labels before making targets

The `.atr` file is not simply a binary “arrhythmia / normal” label. It contains beat locations and beat symbols, plus rhythm changes, signal quality, and comments. Common beat symbols include `N` normal, `V` PVC, `A` atrial premature, `L`/`R` bundle branch block, `/` paced, and `Q` unclassifiable. Rhythm labels are carried in auxiliary text such as `(AFIB`, `(VT`, and `(N`; `~` marks a signal-quality change. Consult the database annotation table rather than infer meanings from the symbol alone. Expert-reviewed beat labels are generally aligned near the R-wave peak, with documented exceptions/updates. [MIT-BIH annotation table](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm) [annotation format and rhythm strings](https://archive.physionet.org/physiobank/annotations.shtml)

Choose a task explicitly:

- **QRS detection:** find beat times; evaluate event matching within a stated tolerance.
- **Beat classification:** classify a window around each reference beat; say which symbol groups/classes are included.
- **Rhythm classification:** classify sequences over seconds/minutes; use rhythm interval labels, not a beat's morphology alone.
- **Signal quality/artifact detection:** identify unusable/noisy periods; use quality annotations where available, but remember they are human subjective annotations, not a complete taxonomy of motion, electrode, baseline, muscle, and mains artifacts.

Keep these outputs separate. An artifact detector can flag low confidence and suppress an arrhythmia decision; “artifact” itself should not be silently relabeled as a cardiac rhythm.

## 4. DSP foundations and safe filtering

Sampling at `fs=360 Hz` means sample `n` occurs at `t=n/fs`; Nyquist is `fs/2=180 Hz`. A digital filter is a recurrence. For an IIR filter:

`y[n] = Σ(b[k] x[n-k]) - Σ(a[k] y[n-k])`, with `a[0]` normalized to 1.

The existing `dsp_processing.py` uses a third-order Butterworth bandpass (0.5–40 Hz) and `filtfilt`. This is good for an offline visualization experiment: Butterworth is smooth in the passband and forward-backward filtering has zero phase. It is not realtime because `filtfilt` uses future samples. Also, a 40 Hz low-pass can attenuate useful sharp morphology, depending on the task; filter choice must follow the endpoint and be reported. Do not apply a 50/60 Hz notch automatically: MIT-BIH was analog-filtered to about 0.1–100 Hz before digitization and has recorder/playback artifacts; not all narrowband energy is mains interference. [digitization notes](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)

For streaming, use a causal filter and preserve its state across chunks. Prefer second-order sections to reduce numerical problems:

```python
from scipy import signal

sos = signal.butter(3, [0.5, 40], btype='bandpass', fs=fs, output='sos')
zi = signal.sosfilt_zi(sos) * x[0, 0]   # initialize once per stream/lead
y, zi = signal.sosfilt(sos, chunk[:, 0], zi=zi)
# For the next chunk, pass the returned zi. Do not reset filter state each chunk.
```

Filtering tradeoffs to learn and measure: passband/stopband, transition width, ripple, phase/group delay, order, transient settling, quantization, and compute cost. A high-pass attenuates baseline wander but can distort ST segments; a low-pass suppresses EMG/high-frequency noise but can blunt QRS edges; a notch can distort nearby morphology. Always compare raw and filtered traces and PSDs, then validate detection on the chosen filtered signal. Document the exact coefficients and sample rate because coefficients depend on `fs`.

## 5. A transparent QRS detector baseline

Pan–Tompkins-style processing makes the QRS slope prominent: bandpass → derivative → square → moving integration → adaptive threshold → refractory/peak localization. With `x[n]` filtered, a simple derivative is `d[n]=x[n]-x[n-1]`; squared energy is `q[n]=d[n]^2`; moving integration over `M≈0.10–0.15 fs` is `e[n]=(1/M)Σ(q[n-k])`. This is a feature envelope, not the R-wave itself. Detect a candidate on `e`, then refine its timestamp to a local maximum/minimum in the filtered ECG within a bounded neighborhood. Maintain separate signal/noise estimates and adaptive thresholds; enforce a physiologic refractory interval; optionally search back for a missed beat after an unusually long RR. Store detector state across chunks.

Do not use the current `realtime_detection.py` as an accuracy result. It repeatedly filters an overlapping 2-second list, which resets IIR state and reprocesses past samples; then it thresholds one squared derivative sample without the moving integration/local-maximum logic. Its threshold is updated before comparison (making the comparison partly self-referential), the first detected interval has no valid previous beat, and `if rr:` can hide a zero/invalid value. Those choices can produce plausible plots without valid sensitivity or timing. First implement and evaluate offline on whole records, then use the *same causal stateful algorithm* in the streaming simulator.

## 6. Artifact detection and extraction

Start by **detecting and characterizing** artifacts, not blindly removing them. Common ECG contamination includes baseline wander (respiration/motion/electrode drift), power-line interference, EMG, abrupt electrode motion, saturation/clipping, lead-off/flatline, and transient impulses. For each type, ask: what frequency/time pattern distinguishes it from cardiac morphology, what reference label exists, and what cardiac content could the proposed filter remove?

Create window-level feature tables and compare clean/noisy annotations. Candidate features include robust range, RMS, derivative RMS, kurtosis, line-band power ratio, low-frequency/high-frequency power ratios, flatline fraction, clipping fraction, QRS detector confidence, and cross-lead agreement. Set thresholds from training records only, then calibrate on held-out records. If you want actual artifact waveforms, define a clean reference and an extraction method (e.g., baseline estimate by low-pass/spline; residual `artifact ≈ raw − clean_estimate` under explicit assumptions). This residual is not ground truth: ECG and noise overlap in time and frequency. Wavelets, adaptive filters, and ICA can help in settings with suitable reference channels, but they introduce assumptions and compute/memory costs. The MIT-BIH Noise Stress Test Database provides calibrated noise stress tests for algorithm evaluation. [PhysioNet related database](https://physionet.org/content/mitdb/1.0.0/)

## 7. Evaluation without leakage

Split by **record/person**, never randomly by beat window. Adjacent windows from one recording are highly correlated and leaking them across train and test makes results optimistic. Keep 201 and 202 together because they belong to one person. Use a development split for choices, a validation split for thresholds, and a locked record-level test for final reporting. Since 25 recordings were intentionally selected for uncommon arrhythmias, report per-class results and avoid interpreting pooled proportions as prevalence. [PhysioNet usage notes](https://physionet.org/content/mitdb/1.0.0/)

For QRS detection, match each detected event to at most one reference event within a declared tolerance. Report `Se=TP/(TP+FN)`, `PPV=TP/(TP+FP)`, `F1=2·Se·PPV/(Se+PPV)`, timing error, false positives/hour, and per-record results. For class prediction also report confusion matrix, class-wise sensitivity/precision/F1, and support; overall accuracy is misleading under imbalance. For signal quality report sensitivity and specificity for noisy/unreadable segments, event duration overlap, and per-type breakdown if labels permit. State any excluded records and the exact symbol-to-class mapping. Never tune preprocessing, thresholds, or class mapping on the locked test records.

## 8. From Python prototype to edge deployment

Build an incremental interface such as `process(samples) -> events, quality, state`. Test it with chunk sizes 1, 16, 64, and irregular chunk boundaries; outputs should be invariant to chunking apart from documented buffering latency. Maintain filter delay state, integration ring buffer/sum, recent RR history, adaptive signal/noise estimates, refractory counter, and timestamp counter. Use a fixed ring buffer rather than a Python list with `pop(0)`; process each sample once.

Before deployment, specify target MCU/CPU, RAM, flash, sample format, channel count, maximum latency, power budget, and failure behavior. Measure worst-case execution time and memory with realistic data. Then port coefficients and state updates, consider fixed-point quantization with overflow/saturation tests, compare embedded outputs against Python golden vectors, and run long streams, lead dropouts, clipping, noise, and reset/reconnect cases. For an ML model, quantize only after a record-wise baseline and measure calibration and per-class performance again. An edge detector needs an explicit “signal quality insufficient / uncertain” state.

## 9. A practical learning order

1. Run `data_exp.py`; verify units, timing, and both lead names from headers.
2. Expand `eda_detailed.py` to inventory all records, annotations, and per-record statistics; inspect raw strips around unusual labels.
3. Compare raw and offline filtered signal, PSD, and spectrogram; explain each filter change.
4. Implement offline QRS detection and event matching; debug missed/extra peaks by plotting examples.
5. Add rhythm/beat tasks and signal-quality detection as distinct problems.
6. Make the detector causal and stateful, then compare chunked streaming against the offline causal pass.
7. Evaluate once on held-out people/records, report class-wise metrics and latency, and only then optimize for the target edge device.

The highest-value next code change is replacing the current realtime demo with a stateful causal detector and a proper annotation-matching evaluation. Keep the current figures as illustrations, not performance evidence.
