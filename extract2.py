import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file
audio_file = 'audio.wav'
y, sr = librosa.load(audio_file)

# Compute MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Display MFCC
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()



import numpy as np
import librosa.display

# Load audio file
y, sr = librosa.load('audio.wav')

# Compute the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Display spectrogram
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Compute Zero Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y)

# Plot Zero Crossing Rate
plt.plot(zcr.T)
plt.xlabel('Frames')
plt.ylabel('Zero Crossing Rate')
plt.title('Zero Crossing Rate')
plt.show()


# Compute chroma feature
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Display Chroma Feature
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chroma Feature')
plt.show()
