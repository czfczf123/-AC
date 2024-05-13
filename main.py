import torch

from TTS.api import TTS
import os
#import TTS.tts.models.vits
torch.cuda.set_device(0)

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one
#model_name = TTS.list_models()[0]
# Init TTS

# Example voice cloning with YourTTS in English, French and Portuguese:



model_path="YourTTS-EN-VCTK-May-13-2024_01+49PM-0000000/best_model.pth"
config_path="YourTTS-EN-VCTK-May-13-2024_01+49PM-0000000/config.json"
src="test_wav/czf.wav"
src_wav_ch="../datasets/aishell1/wav/S0002/BAC009S0002W0127.wav"
tar_wav="test_wav/四川话.wav"
src_wav_en="../datasets/VCTK/wav48_silence_trimmed/p229/p229_004_mic1.flac"
src_wav_ja="../datasets/jvs/jvs002/nonpara30/wav24kHz16bit/BASIC5000_0338.wav"

tts = TTS(model_path=model_path,
          config_path=config_path,
          progress_bar=False, gpu=True)

tts.vc_to_file(source_wav=tar_wav,target_wav=tar_wav,file_path="output.wav")

#tts.tts_to_file(text="数你表现好呢",speaker_wav=tar_wav,language="ch",file_path="output_TTS.wav")

#tts.tts_to_file("you are beautiful", speaker_wav=r"C:\Users\程游侠\Desktop\czfo.wav", file_path="output.wav")
#tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
#tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
