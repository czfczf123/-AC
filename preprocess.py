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
kespeech_config=BaseDatasetConfig(
    formatter="kespeech1",
    dataset_name="kespeech",
    meta_file_train="",
    meta_file_val="",
    path="../datasets/KeSpeech",
    language="mul",
    phonemizer="",
)
# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to added new datasets just added they here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [jvs_config]


def get_bert(text,language):
    if language=="ch":
        checkpoint ="bert/bert-base-chinese"
    elif language=="en":
        checkpoint ="bert/bert-base-cased"
    elif language=="ja":
        checkpoint="bert/bert-base-japanese"
    else:
        raise "数据集不知道是啥语音"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(text, return_tensors="pt")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    res = model(**inputs, output_hidden_states=True).hidden_states
    res = torch.cat(res[-3:-2], -1)[0].cpu()
    return res


device="cuda:0"
hmodel = utils.get_hubert_model().to(device)
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
    speaker_mapping = {}

    class_name_key = encoder_manager.encoder_config.class_name_key

    if dataset_conf.language == "ch":
        checkpoint = "bert/bert-base-chinese"
    elif dataset_conf.language == "en":
        checkpoint = "bert/bert-base-cased"
    elif dataset_conf.language == "ja":
        checkpoint = "bert/bert-base-japanese"
    else:
        raise "数据集不知道是啥语音"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)

    for fields in tqdm(samples):
        class_name = fields[class_name_key]
        audio_file = fields["audio_file"]
        filename = os.path.basename(audio_file)
        filename = filename.split(".")[0]
        speaker_name = fields["speaker_name"]
        embedding_key = fields["audio_unique_name"]

        wav, sr = librosa.load(audio_file, sr=16000)
        #处理基频：
        speaker_path = os.path.join(f0_path, speaker_name)
        if os.path.exists(speaker_path):
            pass
        else:
            os.mkdir(speaker_path)
        path1 = os.path.join(speaker_path, filename + ".f0.npy")
        if os.path.exists(path1):
            pass
        else:
            f0 = compute_pitch_pyworld(wav, 16000, 320)
            f0[f0<0]=0
            np.save(path1,f0)
        #处理speaker embedding
        embedd = encoder_manager.compute_embedding_from_clip(audio_file)
        speaker_mapping[embedding_key] = {}
        speaker_mapping[embedding_key]["name"] = class_name
        speaker_mapping[embedding_key]["embedding"] = embedd

        #处理hubert
        speaker_path = os.path.join(hubert_path, speaker_name)
        if os.path.exists(speaker_path):
            pass
        else:
            os.mkdir(speaker_path)
        path2=os.path.join(speaker_path,filename + ".soft.pt")
        if os.path.exists(path2):
            pass
        else:
            c_embedd = compute_hubert(audio_file, hmodel)
            torch.save(c_embedd, path2)
        #处理Bert
        speaker_path = os.path.join(bert_path, speaker_name)
        if os.path.exists(speaker_path):
            pass
        else:
            os.mkdir(speaker_path)
        path3 = os.path.join(speaker_path, filename + ".bert.pt")
        if os.path.exists(path3):
            pass
        else:
            inputs = tokenizer(fields["text"], return_tensors="pt")
            res = model(**inputs, output_hidden_states=True).hidden_states
            res = torch.cat(res[-3:-2], -1)[0].cpu()
            torch.save(res,path3)

    mapping_file_path=os.path.join(dataset_conf.path,"speakers.pth")
    save_file(speaker_mapping, mapping_file_path)



