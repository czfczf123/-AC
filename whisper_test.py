import torch
import whisper
torch.cuda.set_device(3)
model = whisper.load_model("medium")
result = model.transcribe("test_wav/czf.wav")
print(result["text"])



# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("test_wav/czf.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)