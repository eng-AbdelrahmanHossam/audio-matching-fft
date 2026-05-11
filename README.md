# Audio Matching Using Fourier Transform (FFT)

A simplified Shazam-like system that locates a short audio clip within a longer recording using frequency-domain analysis.

## Overview

This project implements a sliding-window matching algorithm that uses the **Fast Fourier Transform (FFT)** to compare the spectral patterns of audio signals. It extracts a query clip from a full recording, converts both to the frequency domain, and finds the clip's position using cosine similarity.

## How It Works

1. **Load & Preprocess** - Read `.wav` file, convert stereo to mono
2. **Extract Query Clip** - Select a segment from the full signal by specifying start time and duration
3. **Frequency Domain** - Compute one-sided FFT magnitude spectrum
4. **Sliding Window Matching** - Slide a window across the full signal, compare FFTs using normalized dot product
5. **Detect Best Match** - Find the window with highest similarity score
6. **Visualize Results** - Plot time domain, frequency domain, similarity scores, and comparison

## Requirements

- Python 3.x
- numpy
- matplotlib
- scipy

## Setup

```bash
pip install -r requirements.txt