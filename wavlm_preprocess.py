import os

import torch
from tqdm import tqdm
import torchaudio
import utils
from TTS.bin.compute_wavlm import compute_wavlm
from TTS.config import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager
from wavlm.WavLM import WavLM, WavLMConfig
dataset_conf=r"../datasets/sishell1"
mapping_file_path = os.path.join(dataset_conf, "speakers1.pth")
embeddings_file = os.path.join(dataset_conf, "speakers.pth")
c_dataset = BaseDatasetConfig()
c_dataset.formatter = "aishell1"
c_dataset.dataset_name = "aishell1"
c_dataset.path ="../datasets/aishell1"
meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=True,eval_split_size=0.01)
samples = meta_data_train + meta_data_eval
print("读取训练姐完成")
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "checkpoints/model_se.pth"
)
SPEAKER_ENCODER_CONFIG_PATH ="checkpoints/config_se.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('wavlm/path/to/WavLM-Base+.pt',map_location="cuda:3")
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])

print("hubert初始化完成")
#os.mkdir(os.path.join(dataset_conf,"hubert_results"))
print("开始")
for fields in tqdm(samples):
    audio_file = fields["audio_file"]
    filename = os.path.basename(audio_file)
    filename=filename.split(".")[0]
    speaker_name=fields["speaker_name"]
    path = os.path.join(fields["root_path"], "wavlm_results",speaker_name)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    path1 = os.path.join(path, filename + ".soft.pt")
    wav_input_16khz, sr = torchaudio.load(audio_file)
    if sr!=16000:
        raise "采样率不是16000"

    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
    rep = model.extract_features(wav_input_16khz)[0]
    torch.save(rep,path1)
print("完成")