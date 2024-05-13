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
from utils import interpolate_f0
from TTS.tts.utils.speakers import SpeakerManager
from transformers import AutoTokenizer, AutoModelForMaskedLM
from TTS.tts.utils.managers import save_file
from utils import f0_to_coarse
vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="vctk",
    meta_file_train="",
    meta_file_val="",
    path="../datasets/VCTK",
    language="en",
    ignored_speakers=[
        "p261",
        "p225",
        "p294",
        "p347",
        "p238",
        "p234",
        "p248",
        "p335",
        "p245",
        "p326",
        "p302",
    ],  # Ignore the test speakers to full replicate the paper experiment
    phonemizer=""
)

aishell1_config=BaseDatasetConfig(
    formatter="aishell1",
    dataset_name="aishell1",
    meta_file_train="",
    meta_file_val="",
    path="../datasets/aishell1",
    language="ch",
    phonemizer="",
)

jvs_config=BaseDatasetConfig(
    formatter="jvs",
    dataset_name="jvs",
    meta_file_train="",
    meta_file_val="",
    path="../datasets/jvs",
    language="ja",
    phonemizer="",
)
# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to added new datasets just added they here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [jvs_config]




use_cuda = torch.cuda.is_available()
SPEAKER_ENCODER_CHECKPOINT_PATH = "checkpoints/model_se.pth"
SPEAKER_ENCODER_CONFIG_PATH = "checkpoints/config_se.json"
encoder_manager = SpeakerManager(
        encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
        encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
        d_vectors_file_path=None,
        use_cuda=use_cuda,
    )
for dataset_conf in DATASETS_CONFIG_LIST:
    meta_data_train, meta_data_eval = load_tts_samples(dataset_conf, eval_split=True, eval_split_size=0.01)
    samples = meta_data_train + meta_data_eval

    f0_path=os.path.join(dataset_conf.path,"f0_results")
    hubert_path=os.path.join(dataset_conf.path,"hubert_results")
    bert_path=os.path.join(dataset_conf.path,"bert_results")


    class_name_key = encoder_manager.encoder_config.class_name_key


    for fields in tqdm(samples):
        class_name = fields[class_name_key]
        audio_file = fields["audio_file"]
        filename = os.path.basename(audio_file)
        filename = filename.split(".")[0]
        speaker_name = fields["speaker_name"]
        embedding_key = fields["audio_unique_name"]

        wav, sr = librosa.load(audio_file, sr=16000)
        if wav.shape[0]>16000*15:
            print("超出15s")
            print(wav.shape)

        #处理基频：
        speaker_path = os.path.join(f0_path, speaker_name)
        if os.path.exists(speaker_path):
            pass
        else:
            os.mkdir(speaker_path)
        path1 = os.path.join(speaker_path, filename + ".f0.npy")
        if os.path.exists(path1):
            f0=np.load(path1)
            f0=torch.from_numpy(f0)
            if torch.isnan(f0).any():
                print("有f0傻逼")
            f0 = f0_to_coarse(f0)
            if torch.isnan(f0).any():
                print("有coarse傻逼")



        #处理hubert
        speaker_path = os.path.join(hubert_path, speaker_name)
        if os.path.exists(speaker_path):
            pass
        else:
            os.mkdir(speaker_path)
        path2=os.path.join(speaker_path,filename + ".soft.pt")
        if os.path.exists(path2):
            hubert=torch.load(path2)
            if torch.isnan(hubert).any():
                print("而傻逼")





