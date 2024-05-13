import os

import torch
from tqdm import tqdm

import utils
from TTS.bin.compute_hubert import compute_hubert
from TTS.config import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager
dataset_conf=r"../datasets/VCTK"
mapping_file_path = os.path.join(dataset_conf, "speakers1.pth")
embeddings_file = os.path.join(dataset_conf, "speakers.pth")
c_dataset = BaseDatasetConfig()
c_dataset.formatter = "vctk"
c_dataset.dataset_name = "vctk"
c_dataset.path ="../datasets/VCTK"
meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=True,eval_split_size=0.01)
samples = meta_data_train + meta_data_eval
print("读取训练姐完成")
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "checkpoints/model_se.pth"
)
SPEAKER_ENCODER_CONFIG_PATH ="checkpoints/config_se.json"
encoder_manager = SpeakerManager(
        encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
        encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    )
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
hmodel = utils.get_hubert_model().to(device)
print("hubert初始化完成")
class_name_key = encoder_manager.encoder_config.class_name_key
#os.mkdir(os.path.join(dataset_conf,"hubert_results"))
print("开始")
for fields in tqdm(samples):
    class_name = fields[class_name_key]
    audio_file = fields["audio_file"]
    filename = os.path.basename(audio_file)
    filename=filename.split(".")[0]
    speaker_name=fields["speaker_name"]
    path = os.path.join(fields["root_path"], "hubert_results",speaker_name)
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    path1 = os.path.join(path, filename + ".soft.pt")
    c_embedd = compute_hubert(audio_file, hmodel)
    torch.save(c_embedd,path1)
print("完成")


