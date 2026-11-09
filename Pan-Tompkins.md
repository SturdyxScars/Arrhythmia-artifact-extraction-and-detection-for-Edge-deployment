The key difference is that P waves and T waves represent slower electrical activity, while the QRS complex (especially the R peak) represents very rapid ventricular depolarization.

ECG Components

A normal heartbeat looks like:

      P          T
     /\         /\
    /  \       /  \
___/    \__/\_/    \____
            QRS

P Wave

Represents:

Atrial depolarization (electrical activation of the atria)

Characteristics:

Small amplitude
Smooth rounded shape
Slow rising and falling edges
Lower frequency content (≈ 0.5–10 Hz)

Example:

   /\
  /  \
 /    \

QRS Complex

Represents:

Ventricular depolarization (electrical activation of the ventricles)

Characteristics:

Highest amplitude
Very steep slopes
Short duration (typically 80-120 ms)
Frequency content concentrated around 5-15 Hz

Example:

   /\     ← R peak
__/  \__


The R peak is usually the largest feature in the ECG.

T Wave

Represents:

Ventricular repolarization (recovery phase)

Characteristics:

Broader than QRS
Smooth and rounded
Lower slope
Usually larger than P wave but much wider

Example:

      /\
     /  \
 ___/    \___

Why Pan-Tompkins Can Separate Them
1. Slope Difference

Look at the slope:

P Wave
      /\
     /  \


Slope changes gradually.

Derivative output:

   small

QRS Complex
      /\
_____/  \____


Slope changes very rapidly.

Derivative output:

     HIGH


This is why the derivative stage highlights QRS much more than P or T.

2. Frequency Difference

Approximate frequency ranges:

Wave	Dominant FrequencyP wave	0.5-10 Hz
T wave	1-7 Hz
QRS	5-15 Hz

When Pan-Tompkins applies a 5-15 Hz bandpass filter, it:

preserves most QRS energy
suppresses much of the P wave
suppresses much of the T wave
removes baseline wander

So after filtering:

P  → smaller
T  → smaller
QRS → prominent

3. Squaring Enhancement

Suppose after derivative:

Wave	Derivative MagnitudeP	2
T	3
QRS	10

After squaring:

Wave	SquaredP	4
T	9
QRS	100

Notice how QRS becomes overwhelmingly dominant.

This is one reason squaring is so effective.

4. Width Difference

Typical durations:

Wave	DurationP	80-100 ms
QRS	80-120 ms
T	160-250 ms

T waves are wider and smoother.

After moving-window integration:

QRS --> sharp energy pulse
T   --> broad low pulse


Thresholding can distinguish these shapes.

In Arrhythmia ECGs from MIT-BIH

Things become interesting because abnormal beats change QRS morphology.

Normal Beat
     /\
____/  \____

PVC (Premature Ventricular Contraction)
      /\
     /  \
____/    \____

Wider QRS
Larger energy
Different shape

Even though morphology changes, the QRS still usually has:

✅ steep slopes

✅ high energy

✅ dominant frequency content

Therefore Pan-Tompkins can still detect the R peak effectively.

A DSP Engineer's View

Think of it this way:

Feature	P Wave	T Wave	QRSAmplitude	Low	Medium	High
Slope	Low	Low	Very High
Energy	Low	Medium	Very High
Width	Medium	Wide	Narrow
Frequency	Low	Low	Higher
Derivative Response	Weak	Weak	Strong
Squaring Response	Weak	Weak	Dominant

This is exactly why the sequence

Bandpass
→ Derivative
→ Squaring
→ Moving Average


works so well on MIT-BIH ECG recordings. Each stage progressively suppresses P and T waves while enhancing the QRS complex, making the R peak stand out clearly for heartbeat and arrhythmia analysis.