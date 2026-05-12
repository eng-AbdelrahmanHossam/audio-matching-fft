# Audio Matching Using Fourier Transform (FFT)

A simplified Shazam-like system that locates a short audio clip within a longer recording using frequency-domain analysis.

## Overview

This project implements a sliding-window matching algorithm that uses the **Fast Fourier Transform (FFT)** to compare the spectral patterns of audio signals. It loads a separate query clip file and finds its position within a longer recording using cosine similarity.

## How It Works

1. **Load & Preprocess** - Read two `.wav` files (full recording and query clip), convert stereo to mono
2. **Frequency Domain** - Compute one-sided FFT magnitude spectrum for the query clip
3. **Sliding Window Matching** - Slide a window across the full signal, compare FFTs using normalized dot product
4. **Detect Best Match** - Find the window with highest similarity score
5. **Visualize Results** - Plot time domain, frequency domain, similarity scores, and comparison

## Files Required

- `sample.wav` - The full audio recording to search within
- `clip.wav` - The audio clip you want to locate (must be shorter than sample.wav)

## Requirements

- Python 3.x
- numpy
- matplotlib
- scipy

## Audio Files

This repository does **not** include audio files due to copyright. To use this code:

1. Provide your own `sample.wav` (full audio recording)
2. Provide your own `clip.wav` (audio clip to find)
3. Both files should be placed in the same directory as the Python script

## Setup

```bash
pip install -r requirements.txt