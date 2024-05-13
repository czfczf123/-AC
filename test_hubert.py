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
jvs_config=BaseDatasetConfig(
    formatter="jvs",
    dataset_name="jvs",
    meta_file_train="",
    meta_file_val="",
    path="../datasets/jvs",
    language="ja",
    phonemizer="",
)

DATASETS_CONFIG_LIST = [jvs_config]
meta_data_train, meta_data_eval = load_tts_samples(jvs_config, eval_split=True, eval_split_size=0.01)
samples = meta_data_train + meta_data_eval
print(samples[:8])