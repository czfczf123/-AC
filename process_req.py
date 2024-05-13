import os

import librosa
import numpy as np
import torch
from tqdm import tqdm

import utils
from TTS.bin.compute_hubert import compute_hubert
from TTS.config import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.vits.naturalspeech2_pytorch import compute_pitch_pyworld
from repcodec.examples.whisper_feature_reader import WhisperFeatureReader
from utils import interpolate_f0
from TTS.tts.utils.speakers import SpeakerManager
from transformers import AutoTokenizer, AutoModelForMaskedLM
from TTS.tts.utils.managers import save_file



kespeech_config=BaseDatasetConfig(
    formatter="kespeech1",
    dataset_name="kespeech",
    meta_file_train="",
    meta_file_val="",
    path="/media/D/czf/KeSpeech",
    language="mul",
    phonemizer="",
)
# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to added new datasets just added they here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [kespeech_config]


device="cuda:3"
reader = WhisperFeatureReader(root=None,ckpt="medium", layer=24, device=device)
use_cuda = torch.cuda.is_available()
for dataset_conf in DATASETS_CONFIG_LIST:
    meta_data_train, meta_data_eval = load_tts_samples(dataset_conf, eval_split=True, eval_split_size=0.01)
    samples = meta_data_train + meta_data_eval

    f0_path=os.path.join(dataset_conf.path,"f0_results")
    hubert_path=os.path.join(dataset_conf.path,"hubert_results")
    bert_path=os.path.join(dataset_conf.path,"bert_results")
    whisper_path = os.path.join(dataset_conf.path, "whisper_jz_results")

    if os.path.exists(hubert_path):
        pass
    else:
        os.mkdir(hubert_path)
    if os.path.exists(bert_path):
        pass
    else:
        os.mkdir(bert_path)
    if os.path.exists(f0_path):
        pass
    else:
        os.mkdir(f0_path)
    if os.path.exists(whisper_path):
        pass
    else:
        os.mkdir(whisper_path)
    speaker_mapping = {}

    for fields in tqdm(samples):
        audio_file = fields["audio_file"]
        filename = os.path.basename(audio_file)
        filename = filename.split(".")[0]
        speaker_name = fields["speaker_name"]
        embedding_key = fields["audio_unique_name"]

        #处理whisper
        whisper=os.path.join(whisper_path, speaker_name)
        if os.path.exists(whisper):
            pass
        else:
            os.mkdir(whisper)
        path2=os.path.join(whisper, filename + ".whisper.pt")

        if os.path.exists(path2):
            pass
        else:
            try:
                tmp=reader.get_feats(audio_file)
                torch.save(tmp,path2)
            except:
                print(fields)
                exit(99)

