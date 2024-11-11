# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
import matplotlib.pyplot as plt
import numpy as np
from pequegrad.signal import fft
from pequegrad.tensor import Tensor

# sampling rate
sr = 128
# sampling interval
ts = 1.0 / sr
t = np.arange(0, 1, ts)

freq = 1.0
x = 3 * np.sin(2 * np.pi * freq * t)

freq = 4
x += np.sin(2 * np.pi * freq * t)

freq = 7
x += 0.5 * np.sin(2 * np.pi * freq * t)

plt.figure(figsize=(8, 6))
plt.plot(t, x, "r")
plt.ylabel("Amplitude")


x = Tensor(x)

X = fft(x).numpy()

# calculate the frequency
N = X.shape[0]
n = np.arange(N)
T = N / sr
freq = n / T

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.stem(freq, abs(X), "b", markerfmt=" ", basefmt="-b")
plt.xlabel("Freq (Hz)")
plt.ylabel("FFT Amplitude |X(freq)|")

# Get the one-sided specturm
n_oneside = N // 2
# get the one side frequency
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_oneside = X[:n_oneside] / n_oneside


plt.subplot(122)
plt.stem(f_oneside, abs(X_oneside), "b", markerfmt=" ", basefmt="-b")
plt.xlabel("Freq (Hz)")
plt.ylabel("Normalized FFT Amplitude |X(freq)|")
plt.tight_layout()
plt.show()
