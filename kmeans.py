import os
import sys

import librosa
from tqdm import tqdm

from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.config import load_config, BaseDatasetConfig
import torch
from TTS.utils.io import load_fsspec
import psutil

from repcodec.examples.whisper_feature_reader import WhisperFeatureReader

# 获取当前进程的内存信息
process = psutil.Process()

from random import sample
import numpy as np
if torch.cuda.is_available():
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')
'''
model_path="Fineturn-CH-AISHELL1-January-31-2024_11+17AM-0000000/checkpoint_85000.pth"
config_path="Fineturn-CH-AISHELL1-January-31-2024_11+17AM-0000000/config.json"
tts_config = load_config(config_path)
model = Vits.init_from_config(tts_config)
state_dict = torch.load(model_path, map_location="cpu")

#model.load_state_dict(selected_params, strict=False)
model.load_state_dict(state_dict["model"], strict=True)


device = torch.device("cpu")
model.to(device)
model.eval()
'''


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

def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    # convert to float
    X = X.float()
    X1 = X.to("cpu")

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    initial_state_pre = initial_state.clone()
    initial_state = initial_state.to(device)
    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')

    choice_cluster = torch.zeros([X.shape[0]])
    a = X.shape[0]
    b = initial_state.shape[0]
    results = torch.zeros([a, b]).to(device)
    dis = torch.zeros_like(X).to(device)
    A = torch.zeros([a, 192]).to(device)
    print("循环前：")
    allocated_memory = torch.cuda.memory_allocated(device)
    print(f"Allocated GPU memory: {allocated_memory / 1e9:.2f} GB")
    memory_info = process.memory_info()
    print("Memory usage: {} MB".format(memory_info.rss / (1024 ** 2)))
    while True:
        memory_info = process.memory_info()
        print("Memory usage: {} MB".format(memory_info.rss / (1024 ** 2)))


        for j in range(0, b):
            A[:, :] = initial_state[j].repeat(a, 1)

            dis[:, :] = X - A

            dis[:, :] = dis ** 2.0

            results[:, j] = dis.sum(dim=-1).squeeze()
        choice_cluster[:] = torch.argmin(results, dim=1)
        initial_state_pre[:, :] = initial_state[:, :]

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X1, 0, selected)

            initial_state[index] = selected.mean(dim=0)

            del selected

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()


        if center_shift ** 2 < tol:
            break
        del center_shift
    initial_state = initial_state.cpu()

    return choice_cluster.cpu(), initial_state


def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    #torch.cuda.set_device(2)

    print("初始化KMEANS：")
    return initial_state


DATASETS_CONFIG_LIST = []



languages=["Northeastern","Mandarin","Southwestern","Jiao-Liao","Beijing","Zhongyuan","Ji-Lu","Lan-Yin","Jiang-Huai"]
'''
for language in languages:
    i = 0
    meta_data_train, meta_data_eval = load_tts_samples(kespeech_config, eval_split=True, eval_split_size=0.01)
    samples = meta_data_train + meta_data_eval
    samples=[i for i in samples if i["dialect"] ==language]
    #samples =meta_data_eval
    print(language)
    print(len(samples))
    samples=sample(samples,len(samples) if len(samples)<2000 else 2000)
    path=os.path.join(kespeech_config.path,"whisper_results")
    #samples=[i for i in samples if i["audio_file"] =="/media/D/czf/KeSpeech/Audio/1010481/phase1/1010481_4ee86ae9.wav"]
    for fields in tqdm(samples):
        audio_file = fields["audio_file"]
        filename = os.path.basename(audio_file)
        filename = filename.split(".")[0]
        speaker_name = fields["speaker_name"]
        speaker_path = os.path.join(path, speaker_name)
        path1 = os.path.join(speaker_path, filename + ".whisper.pt")
        try:
            hubert=torch.load(path1,map_location=device)
        except:
            continue
        hubert=hubert.unsqueeze(0).transpose(1,-1)
        c_lengths = torch.LongTensor(len(hubert)).to(device)
        c_lengths[0] = hubert.shape[1]

        y=model.kmeans(hubert,c_lengths)#b t c

        if i==0:
            i=i+1
            x=y.squeeze(0).transpose(0,1).to("cpu")
        else:
            x=torch.cat((x,y.squeeze(0).transpose(0,1).to("cpu")),0)

    torch.save(x,language+".pt")

exit(999)
'''
device = torch.device('cuda:3')
zer=torch.zeros(192)
for language in languages:
    print(language)
    tmp=torch.load(language+".pt")

    j=[]
    for i in range(len(tmp)):
        if (tmp[i]==zer).all():
            pass
        else:
            j.append(i)
    tmp=tmp[j]

    random_indices = torch.randperm(len(tmp))[:300000]
    y = tmp[random_indices]
    del tmp
    y = y.to(device)
    cluster_ids_x, cluster_centers = kmeans(
        X=y, num_clusters=16, tol=0.01, distance='euclidean', device=device
    )
    torch.save(cluster_centers, 'km/'+language+'.pt')

exit(4)

print("所占用内存大小为：")
print(x.shape)
random_indices = torch.randperm(len(x))[:500000]
y = x[random_indices]

del x
print("所占用内存大小为：")
print(y.shape)
device = torch.device('cuda:3')
y=y.to(device)
cluster_ids_x, cluster_centers = kmeans(
    X=y, num_clusters=32, tol=0.01,distance='euclidean', device=device
)

print(cluster_centers)
torch.save(cluster_centers, 'km_all.pt')
print("over")



