import soundfile as sf
import numpy as np
import librosa

#wav,sr=librosa.load("test_wav/zbc.wav")
wav,sr=sf.read("test_wav/zbc.wav")
print(wav.max())
print(wav.min())
print(wav)
ra=np.random.uniform(low=wav.min()/20,high=wav.max()/20,size=wav.shape)

audio_data=wav+ra

sf.write("audio.wav",audio_data,sr,'PCM_16')