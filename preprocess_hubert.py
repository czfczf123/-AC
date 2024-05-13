
from tqdm import tqdm
import json
import utils
from TTS.bin.compute_hubert import compute_hubert
from TTS.config import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

from epitran.backoff import Backoff

from TTS.tts.utils.text.phonemizers import JA_JP_Phonemizer

backoff = Backoff(['jpn-Hrgn', 'eng-Latn', 'cmn-Hans'], cedict_file='g2p/cedict_1_0_ts_utf-8_mdbg.txt')

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
DATASETS_CONFIG_LIST = []

dic=[]

for dataset_conf in DATASETS_CONFIG_LIST:
    meta_data_train, meta_data_eval = load_tts_samples(dataset_conf, eval_split=True, eval_split_size=0.01)
    samples = meta_data_train + meta_data_eval

    for fields in tqdm(samples):
        text=fields["text"]
        yinsu=backoff.trans_list(text)
        for i in yinsu:
            if i in dic:
                pass
            else:
                dic.append(i)


DATASETS_CONFIG_LIST = []

dic=[]

for dataset_conf in DATASETS_CONFIG_LIST:
    meta_data_train, meta_data_eval = load_tts_samples(dataset_conf, eval_split=True, eval_split_size=0.01)
    samples = meta_data_train + meta_data_eval

    for fields in tqdm(samples):
        text=fields["text"]
        yinsu=backoff.trans_list(text)
        for i in yinsu:
            if i in dic:
                pass
            else:
                dic.append(i)


DATASETS_CONFIG_LIST = [jvs_config]

dic=[]
e = JA_JP_Phonemizer()
for dataset_conf in DATASETS_CONFIG_LIST:
    meta_data_train, meta_data_eval = load_tts_samples(dataset_conf, eval_split=True, eval_split_size=0.01)
    samples = meta_data_train + meta_data_eval
    print(samples[2])
    exit()

    for fields in tqdm(samples):
        text=fields["text"]
        yinsu=e.phonemize(text)
        for i in yinsu:
            if i in dic:
                pass
            else:
                dic.append(i)
print(dic)
with open('dict_ja.json', 'w') as f:
    json.dump(dic, f)






