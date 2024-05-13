import librosa

tar_wav="test_wav/czf.wav"
import torchaudio
import soundfile as sf
wav16k, sr = librosa.load(tar_wav)
sf.write(tar_wav, wav16k, sr)
print("完成")

torchaudio.load(tar_wav)