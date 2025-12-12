"""
utils/features.py

Helper functions for MFCC extraction and simple augmentations. Import from scripts when needed.
"""
import numpy as np
import librosa

def extract_mfcc_from_file(path, sr=8000, n_mfcc=20, duration=None):
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if duration is not None and len(y) > int(duration * orig_sr):
        y = y[:int(duration * orig_sr)]
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1).astype(np.float32)

def add_noise(y, snr_db=20.0):
    rms = (y**2).mean()**0.5
    snr = 10**(snr_db / 20.0)
    noise_rms = rms / snr
    noise = np.random.normal(0, noise_rms, size=y.shape)
    return y + noise