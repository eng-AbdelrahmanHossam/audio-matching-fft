import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft


#Load and Preprocess the Audio Signal
fs, signal= wavfile.read("sample.wav")
if len(signal.shape)>1:
    signal = signal.mean(axis=1)
    
signal=signal.astype(np.float32)
print(f"Sampling frequency:{fs} Hz")
print(f"length of full signal: {len(signal)} samples")
print(f"Duration:{len(signal)/fs:.2f} seconds")

#Time domain plotting and query clip extraction
timefull=np.arange(len(signal))/fs
plt.figure(1)
plt.plot(timefull[:2000], signal[:2000])
plt.title('Full Signal (Time Domain) - First Portion')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

clipStart=14
clipLen=4

clipStartSample=int(clipStart*fs)
clipLenSamples=int(clipLen*fs)
clipEndSample=clipStartSample+clipLenSamples

clipSignal=signal[clipStartSample:clipEndSample]
cliptime=np.arange(len(clipSignal))/fs

print(f"Clip length: {len(clipSignal)} samples")
print(f"Clip duration: {len(clipSignal)/fs:.2f} seconds")
print(f"Original clip position: {clipStart} seconds")
plt.figure(2)
plt.plot(cliptime, clipSignal)
plt.title('Query Clip (Time Domain)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

#FFT
def getfft(signalSegment):
    N=len(signalSegment)
    freqData=fft(signalSegment)
    freqData=freqData[:N//2]
    magnitude=(2/N)*np.abs(freqData)
    frequencies=np.linspace(0, fs/2, N//2)
    return frequencies, magnitude
freqClip, clipfft = getfft(clipSignal)

plt.figure(3)
plt.plot(freqClip, clipfft)
plt.title('Query Clip (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

#sliding window
stepSize = int(fs * 0.1)
clipLen = len(clipSignal)
_, clipfft = getfft(clipSignal)

similarityScores = []
positions = []

for start in range(0, len(signal) - clipLen, stepSize):
    window = signal[start:start + clipLen]
    _, windowfft=getfft(window)
    dotP=np.dot(clipfft, windowfft)
    normClip=np.linalg.norm(clipfft)
    normWindow=np.linalg.norm(windowfft)
    similarity=dotP/(normClip*normWindow)
    similarityScores.append(similarity)
    positions.append(start/fs)

similarityScores=np.array(similarityScores)
positions=np.array(positions)
    
#Visualize
bestIndex=np.argmax(similarityScores)
detectedPos=positions[bestIndex]
bestScore=similarityScores[bestIndex]
detectedSample=int(detectedPos*fs)

plt.figure(4)
plt.plot(positions, similarityScores)
plt.axvline(x=clipStart, color='g', linestyle='--', label='Actual Position')
plt.axvline(x=detectedPos, color='r', linestyle='--', label='Detected Position')
plt.title('Similarity Score vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Similarity Score')
plt.legend()
plt.grid(True)

detectedSegment=signal[detectedSample:detectedSample+clipLen]
timeDetected=np.arange(len(detectedSegment))/fs

#Compare original vs detected
plt.figure(5)

# Original
plt.subplot(2, 1, 1)
plt.plot(cliptime, clipSignal)
plt.title('Original Clip')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
#detected
plt.subplot(2, 1, 2)
plt.plot(timeDetected, detectedSegment)
plt.title('Detected Segment')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
print(f"\nDetected position: {detectedPos:.2f} seconds")
print(f"Best similarity score: {bestScore:.4f}")

# Save all figures as images
plt.figure(1)
plt.savefig('full_signal_time.png', dpi=300, bbox_inches='tight')

plt.figure(2)
plt.savefig('clip_time.png', dpi=300, bbox_inches='tight')

plt.figure(3)
plt.savefig('clip_frequency.png', dpi=300, bbox_inches='tight')

plt.figure(4)
plt.savefig('similarity_scores.png', dpi=300, bbox_inches='tight')

plt.figure(5)
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')

plt.show()

plt.show()