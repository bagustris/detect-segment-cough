#!/usr/bin/env python3
"""example for cough segmentation"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../src')
from segmentation import segment_cough
import sounddevice as sd

file = "../sample_recordings/cough.wav"
x, fs = librosa.load(file, sr=None)
plt.plot(x)
plt.show()

# play the audio
sd.play(x, fs)

# segments cough
cough_segments, cough_mask = segment_cough(x,fs, cough_padding=0)
plt.plot(x)
plt.plot(cough_mask)
plt.title("Segmentation Output")
plt.show()

# play segmented cough
sd.play(cough_segments[0], fs)

