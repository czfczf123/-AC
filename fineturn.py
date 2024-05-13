import os
import torchaudio
import torch
from trainer_fineturn.trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
torch.set_num_threads(24)#24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
    In addition, you will need to add the extra datasets following the VCTK as an example.
"""
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

#os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"
#os.environ['LOCAL_RANK']=0

# Name of the run for the Trainer
RUN_NAME = "Fineturn-CH-AISHELL1"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

CONTINUE_PATH="Fineturn-CH-AISHELL1-March-15-2024_01+44PM-0000000"

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH ="YourTTS-EN-VCTK-February-13-2024_10+00AM-0000000/best_model.pth"

# This paramter is usefull to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE =32# 32

# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 16000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 20

### Download VCTK dataset
# Define the number of threads used during the audio resampling
NUM_RESAMPLE_THREADS = 10
# Check if VCTK dataset is not already downloaded, if not download it

#resample_files(VCTK_DOWNLOAD_PATH, SAMPLE_RATE, file_ext="flac", n_jobs=1)

# init configs
kespeech_config=BaseDatasetConfig(
    formatter="kespeech1",
    dataset_name="kespeech",
    meta_file_train="",
    meta_file_val="",
    path="/media/D/czf/KeSpeech",
    language="mul",
    phonemizer="",
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
# Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to added new datasets just added they here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
DATASETS_CONFIG_LIST = [aishell1_config]

### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "checkpoints/model_se.pth"
)
SPEAKER_ENCODER_CONFIG_PATH ="checkpoints/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training
# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it

'''
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_spakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)'''

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=320,#256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that is needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=False,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # On the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Usefull parameters to enable the Speaker Consistency Loss (SCL) discribed in the paper
    # use_speaker_encoder_as_loss=True,
    # Usefull parameters to the enable multilingual training
    use_language_embedding=True,
    embedded_language_dim=4,
    upsample_rates_decoder=[10,8,2,2],
    freeze_hubert_encoder=True,
    freeze_rep=True
)

# General training config, here you can change the batch size and others usefull parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,#8,这个和单线程有关
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=True,
    phonemizer="multi_phonemizer",
    #phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank="#",
        punctuations="!(),-.:;?'",
        phonemes="dict.json",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path="c",
    precompute_num_workers=12,#12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "莫听穿林打叶声，一蓑烟雨任平生",
            "AISHELL1_S0016",
            None,
            "ch",

        ],
        [
            "如果有一天，世界毁灭",
            "AISHELL1_S0017",
            None,
            "ch",

        ],
        [
            "未来会有风雨",
            "AISHELL1_S0058",
            None,
            "ch"

        ],
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets


train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.05#config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)


print("完成模型初始化")
# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(continue_path=CONTINUE_PATH,restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    gpu=3
)
print(trainer)
print("完成训练器初始化")
trainer.fit()
