import os

import librosa
import torch
from tqdm import tqdm

import utils


def compute_hubert(filename,hmodel):
    wav16k, sr = librosa.load(filename, sr=16000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav16k = torch.from_numpy(wav16k).to(device)
    hmodel.to(device)
    c = utils.get_hubert_content(hmodel, wav_16k_tensor=wav16k)
    return c