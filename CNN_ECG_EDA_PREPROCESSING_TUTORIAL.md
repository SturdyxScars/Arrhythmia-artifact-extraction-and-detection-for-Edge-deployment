# ECG EDA and preprocessing for a 1D CNN with streaming inference

This tutorial maps the learning roadmap to the local MIT-BIH files and starter scripts. Work through each stage and keep its outputs: the goal is to understand what every transformation does and to know whether it helps on records the model has never seen.

This is a research/learning project, not a validated diagnostic device. A model trained on MIT-BIH alone is not evidence of clinical performance on modern ambulatory ECG devices or a new patient population.

## 0. Define the prediction problem first

“Arrhythmia detection” can mean different targets. Pick one before preprocessing:

1. **Beat classification:** one label per heartbeat, e.g. AAMI-style N/S/V/F/Q groups. Inputs are fixed windows around beats. This is the simplest 1D CNN starting point.
2. **Rhythm classification:** one label for a multi-second or multi-beat interval, e.g. AF or ventricular tachycardia. Inputs need longer context; a 0.8 s beat crop cannot reliably represent a rhythm episode.
3. **QRS detection:** output event timestamps, not a beat class. This generally needs a signal-processing detector or a dedicated temporal model.
4. **Signal quality/artifact detection:** output usable/noisy/unreadable or artifact type. It is a separate target and may gate the rhythm classifier.

The workflow below starts with **beat classification** and adds streaming. Do not call this a general-purpose arrhythmia alarm: beat classes are not rhythm diagnoses. For your first milestone, focus on `N`, `S`, `V`, `F`, and `Q`, and explicitly state the mapping used from original annotation symbols.

## 1. Know what is in the dataset

MIT-BIH has 48 approximately half-hour two-channel recordings from 47 people. Signals are sampled at 360 Hz per channel. Record 100 in your output has 650,000 samples, so its duration is `650000/360 = 1805.56 s` (about 30.1 minutes); its channels are `MLII` and `V5`. Across records, channel names and orientations vary. Usually the upper channel is modified lead II (MLII); the lower channel is usually modified V1, but some are V2, V4 or V5. Check each header; never infer lead identity solely from array column. Records 201 and 202 are from one person and must stay in the same data split. Twenty-five records were intentionally selected for uncommon arrhythmias, so pooled class proportions are not population prevalence. [PhysioNet MIT-BIH overview](https://physionet.org/content/mitdb/1.0.0/) [record and lead documentation](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)

The matching WFDB files are:

- `.hea`: sampling rate, number of channels, samples, channel names, units/calibration.
- `.dat`: digitized waveform values.
- `.atr`: reference annotations. These include beat labels, but also rhythm, signal-quality and comment annotations. Beat annotation indices are sample positions; time in seconds is `sample/fs`. Rhythm labels often use an auxiliary string; do not treat every symbol as a beat.
- Possible alternate annotation files (for example `102-0.atr`) should not be counted as another record. Select the canonical annotation intentionally and document it.

The MIT-BIH sample values are physically calibrated by WFDB when loading `p_signal`; inspect units and use `d_signal` only when you deliberately need digital counts. Never normalize one entire record before splitting if that lets test-record information affect model parameters; fixed per-window normalization or training-set-only normalization is easier to reason about.

## 2. EDA: build an inventory before plotting examples

Your existing `data_exp.py` is a good first check for record 100, but one record cannot establish dataset-wide properties. Run a metadata pass over all actual headers, and for each record collect:

- ID, `fs`, sample count, duration, channel names and units;
- annotation file chosen and counts of beat symbols, rhythm labels, quality labels and comments;
- missing files, non-finite values, flatline/clipping indicators, per-lead min/max/mean/std and robust spread (median and MAD/IQR);
- beat and target-class counts **per record**, not just one total;
- timestamp range and whether annotations lie within the waveform.

Example skeleton:

```python
from pathlib import Path
from collections import Counter
import numpy as np
import wfdb

root = Path('mit-bih-arrhythmia-database-1.0.0')
record_ids = sorted(p.stem for p in root.glob('*.hea') if p.stem.isdigit())
for rid in record_ids:
    path = root / rid
    rec = wfdb.rdrecord(str(path))
    ann = wfdb.rdann(str(path), 'atr')  # choose canonical annotation explicitly
    x, fs = rec.p_signal, rec.fs
    print(rid, rec.sig_name, rec.units, fs, x.shape,
          f'{x.shape[0]/fs:.1f}s', Counter(ann.symbol))
    for c, lead in enumerate(rec.sig_name):
        v = x[:, c]
        print(' ', lead, 'finite=', np.isfinite(v).all(),
              'min/max=', float(np.min(v)), float(np.max(v)),
              'mean/std=', float(np.mean(v)), float(np.std(v)),
              'median/MAD=', float(np.median(v)),
              float(np.median(np.abs(v-np.median(v)))))
```

Your local folder has 48 `.hea` files and 49 `.atr` files. The extra annotation file is a reminder to enumerate records from headers and choose the intended annotation explicitly, rather than zipping all `.atr` files to signals.

### Visual EDA to perform

For each record, inspect both leads separately:

1. **5–10 second strips:** plot raw ECG with beat annotations. This reveals morphology, lead polarity, baseline, annotation alignment and obvious artifacts.
2. **Longer strips (1–5 minutes):** inspect rhythm changes, drift, intermittent noise and lead dropouts.
3. **Whole record at reduced resolution:** reveal long bad sections and nonstationary behavior. Do not use this downsampled version to train.
4. **Amplitude distribution:** histogram and robust range; look for clipping/flatline, but do not delete unusual values without visual review because abnormal beats can be high amplitude.
5. **RR distribution:** `RR[i]=(sample[i+1]-sample[i])/fs`. Plot histogram and time series, separated by record and rhythm. Exclude non-beat annotations first; indiscriminately differencing all annotation samples corrupts RR.
6. **Welch PSD:** estimate average power by frequency. With `nperseg=4096` at 360 Hz, bin spacing is `360/4096≈0.088 Hz`; PSD indicates average frequency content but not when bursts occur.
7. **Spectrogram:** inspect when noise appears and whether it is sustained or transient. A single PSD can hide time-varying artifact.
8. **Per-record class plots:** class imbalance and morphology differ substantially. Aggregate counts alone hide records with no examples of a class.

Do not assume there is a clean 60 Hz line in every record. MIT-BIH signals were already analog filtered, and the database documentation describes recording/playback artifacts and notes 60 Hz playback contamination; inspect the spectra. It is not a universal preprocessing rule. [MIT-BIH digitization and artifact notes](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)

## 3. Labels and AAMI grouping

The `.atr` symbols describe beats and other events. Common beat symbols include `N` (normal), `L` and `R` (bundle branch block morphologies), `A` (atrial premature), `V` (premature ventricular contraction), `F` (fusion), `/` (paced), and `Q` (unclassifiable). There are additional symbols for escape beats and non-conducted P waves. Rhythm labels may be auxiliary strings such as `(AFIB`; these are intervals/changes in rhythm, not individual beat labels. `~` indicates a signal-quality change. Read the official symbol list and inspect both `ann.symbol` and `ann.aux_note`. [PhysioNet annotation documentation](https://archive.physionet.org/physiobank/annotations.shtml)

A commonly used AAMI beat grouping is:

| Group | Example original symbols | Meaning (coarse) |
|---|---|---|
| N | N, L, R, e, j | Normal and bundle-branch/escape beats |
| S | A, a, J, S | Supraventricular ectopic beats |
| V | V, E | Ventricular ectopic beats |
| F | F | Fusion beats |
| Q | /, f, Q, ? | Paced/fusion-paced/unknown classes (exact mapping varies by protocol) |

This table is a starting convention, not a substitute for your chosen protocol. Publish your exact mapping and exclusions. In particular, decide how you handle paced beats and `?` and do not silently discard classes because they are inconvenient. Rhythm annotations need their own target-building logic.

## 4. Make the split before learning preprocessing parameters

Use a **patient/record-wise split**, never randomly split beat windows. Neighboring beats and repeated morphology from one patient otherwise leak into both train and test. Keep 201 and 202 together because they are the same person. There are published MIT-BIH record splits, but names like DS1/DS2 are not universal: verify the exact record lists and class exclusions in the protocol you choose; do not assume the roadmap's split is automatically “the standard.”

Create three disjoint sets:

- training records: fit normalization statistics, train weights, and perform augmentation;
- validation records: choose hyperparameters, thresholds and stopping point;
- test records: one final locked evaluation.

Any learned transform (mean/std scaling, feature selection, class weights, threshold, synthetic-noise parameters) must be fitted or selected using training/validation only. Avoid balancing by duplicating test examples. Report per-record and per-class numbers because pooled metrics can hide failures on particular people.

## 5. Preprocess a beat-classification dataset

### 5.1 Choose the signal path

For morphology classification, begin with a simple and reproducible path: one named lead (often MLII where present), calibrated mV, causal or documented offline band-pass, then fixed-duration beat windows. Compare single-lead vs two-lead only after establishing a baseline; second-lead identity varies. If using two channels, preserve explicit channel identities or map leads deliberately. Zero-fill/mapping an absent lead as if it were the same view is misleading.

Two common experimental filter choices are:

- **Morphology/display path:** e.g. 0.5–40 Hz, which may preserve broader waveform features but does not guarantee clinical morphology fidelity.
- **QRS detection path:** e.g. 5–15 Hz, chosen to emphasize QRS energy for locating candidate beats, not necessarily to feed the classifier.

Keep the two jobs separate. A narrow detector filter can distort P/T waves and ST information. `filtfilt` in your current `dsp_processing.py` is offline zero-phase processing and cannot be used in causal realtime inference. For deployment, use a causal IIR/FIR and account for its delay. Implement as second-order sections for floating-point stability:

```python
from scipy import signal
sos = signal.butter(3, [0.5, 40], btype='bandpass', fs=fs, output='sos')
y = signal.sosfiltfilt(sos, x)       # offline experiments only
y_stream, zi = signal.sosfilt(sos, chunk, zi=zi)  # causal; carry zi forward
```

Do not train on `sosfiltfilt` windows and claim equivalent causal deployment: phase, edge transients and future context differ. Either train/evaluate using the causal stream path or quantify the mismatch and design the model around it. Inspect filter magnitude response and group delay. Compare filtered and raw strips to verify QRS shape, beat location and boundary behavior.

### 5.2 Find centers and extract windows

For a first beat CNN, use expert annotation positions as the crop centers while studying classification. This isolates beat classification from QRS detector errors. At 360 Hz, a 0.8 s crop has 288 samples. For example, choose 0.25 s before and 0.55 s after the annotated beat (`90` samples before, `198` after); this is asymmetric to include more post-QRS morphology, but test alternatives on validation. A symmetric ±0.4 s crop is also a reasonable baseline. Annotation locations are approximate fiducial points, generally near the R peak; inspect overlays and alignment.

```python
left, right = int(0.25*fs), int(0.55*fs)
window = x[beat_sample-left : beat_sample+right]
```

Drop windows that cross record boundaries; do not wrap around. Decide how to handle windows that overlap noisy periods using signal-quality labels and a documented policy. Keep record ID, lead, sample index, original symbol, grouped target, and quality state alongside each crop. Never lose provenance.

### 5.3 Normalize without erasing useful information

Start with per-window robust centering/scaling, for example subtract the window median and divide by MAD (with a small epsilon), or use a training-set mean/std per lead. Compare against unnormalized mV values. Per-window normalization reduces device gain and baseline differences but can remove absolute amplitude information relevant to some classes. Do not clip or normalize without checking the morphology and amplitude distribution. Fit global statistics on training records only.

### 5.4 Handle imbalance

The number of `N` beats greatly exceeds many abnormal classes. Start with class-weighted cross-entropy computed from training labels, or a carefully controlled weighted sampler. Consider augmentation only on training windows: small amplitude scaling, modest baseline offset/drift, realistic additive noise, and slight time shifts. Avoid transformations that change the clinical label or distort QRS timing/morphology; validate synthetic noise against real traces. Do not make near-identical copies dominate training.

## 6. Build and train a compact 1D CNN

Represent one example as `[channels, samples]` (PyTorch convention) or `[samples, channels]` for many other frameworks. Start with one lead, fixed length and a small model: repeated `Conv1d → normalization → ReLU → pooling`, then global average pooling and a class output. Global pooling reduces parameters relative to flattening a large feature map. Keep a simple baseline (RR + morphology features with a tree model) so you can tell whether the CNN learned useful shape information.

For a batch `B`, logits `z[b,k]` are class scores. Softmax converts them to class probabilities; weighted cross-entropy is `L = -w_y log softmax(z)_y`. Track training and validation loss, per-class sensitivity/precision/F1, confusion matrix, and calibration. Watch the validation curves for overfit. Set the decision threshold using validation records, not the test set. Accuracy alone is a poor metric when classes are imbalanced.

Important limitation: crops centered on reference annotations make a clean supervised classification experiment, but a live device does not know the true beat center. For realistic end-to-end evaluation, first run the streaming QRS detector, classify its predicted beats, and report the combined result. Also report classification conditional on reference-centered beats separately, so the two error sources remain visible.

## 7. Evaluate robustly

Report:

- split record IDs and proof that no patient/record overlaps;
- symbol-to-class mapping, exclusions, channel choice and filters;
- per-class support, sensitivity/recall, precision/PPV, F1 and confusion matrix;
- macro-F1/balanced accuracy and per-record metrics, in addition to overall values;
- QRS detector event sensitivity/PPV and timing error when predictions depend on detected beats;
- inference time, memory and buffering latency for deployment.

For event detection, match each predicted beat at most once to a reference beat within a declared tolerance. For beat-classification scores, a model can look strong on reference-centered crops but fail if the QRS locator misses or shifts beats. Evaluate the full causal pipeline on the locked test records.

## 8. Turn it into realtime inference

“Realtime” means every sample is consumed once, with bounded memory and stated delay. The streaming stages are:

`sample(s) → stateful causal filter → ring buffer → QRS candidate/beat center → crop buffer → CNN → prediction + signal-quality/confidence`

Maintain filter state, sample counter, circular waveform buffer, detector state, recent RR state, and output timestamps. Avoid filtering the entire rolling window again and avoid `list.pop(0)`. The existing `realtime_detection.py` does both repeated rolling-window filtering and a one-sample squared-derivative threshold, so it is not a reliable streaming detector yet. Likewise, your DSP demo's `np.diff` shortens a vector and `mode='same'` convolution is centered/non-causal; both are fine for illustration but need alignment/causal replacements for streaming.

After a candidate QRS is found, wait until the required post-beat samples have arrived before making the crop. For a crop with 0.55 s post-R context, the model has at least 550 ms of intentional buffering latency, plus filter/detector delay and compute. If the application needs a faster output, choose a shorter crop or a different task/model and measure the accuracy tradeoff. Do not hide this delay by plotting the decision at the beat timestamp; report both event time and decision emission time.

Streaming test protocol:

1. Process one sample at a time and in chunks of 16/32/64 samples; results must match within numerical tolerance.
2. Compare chunked output with one continuous causal pass using identical initialization.
3. Test startup, reset, gap, dropped lead, saturation and long noise intervals.
4. Confirm fixed memory and no allocation in the embedded sample loop.
5. Measure worst-case execution time, RAM, flash, average/worst latency and energy on the actual target.
6. Port float32 first and compare embedded outputs with Python golden vectors. Consider int8 quantization only after float inference is evaluated; measure the quantized model again on the same locked test split.

## 9. A sequence of measurable milestones

1. **Data inventory:** all 48 records, headers, channels, annotation counts and class mapping documented.
2. **EDA notebook/script:** representative normal, PVC-heavy, AF/rhythm, noisy and paced examples; raw plots, PSD/spectrogram and RR summaries for both leads.
3. **Crop dataset:** annotation-centered windows, provenance metadata, no boundary wrap, group labels and patient-wise splits.
4. **Baseline classifier:** small 1D CNN compared with a simple feature model; validation-only model selection.
5. **Causal stream:** persistent filters and ring buffer; chunk-size invariance and explicit latency.
6. **End-to-end test:** predicted QRS locations feed the CNN; per-record/per-class reporting on locked records.
7. **Edge profile:** resource use and quantization measured on target hardware.

At each stage, save a small set of plots and example failures. The most useful debugging artifacts are false positives/negatives plotted with the raw trace, filtered trace, annotation, predicted center, crop boundaries and class probabilities.

## Suggested reading

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — dataset description, composition, references and use notes.
- [MIT-BIH database directory](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm) — record selection, lead variation, recording artifacts, annotations and symbols.
- [PhysioBank annotation documentation](https://archive.physionet.org/physiobank/annotations.shtml) — annotation types and auxiliary rhythm labels.
- [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/) — calibrated baseline-wander, muscle-artifact and electrode-motion noise for stress testing. Use it to test robustness; synthetic mixing requires a declared SNR and must not contaminate validation/test procedures.
