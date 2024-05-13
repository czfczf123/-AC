import json
import random

import fsspec
import math
import os
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler
from transformers import BertConfig, BertModel

from AR.models.t2s_model import Text2SemanticDecoder
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder,TextDecoder\
    ,HubertEncoder,SE_fineturn,TransformerCouplingBlock,Hubert_F0_Encoder,F0Decoder,Split,Conv_encoder
from attentions import Decoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask,minmum_path_yinsu_numpy,tanxin_yinsu_numpy
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis   #,fineturn_synthesis
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.utils.io import load_fsspec
from TTS.utils.samplers import BucketBatchSampler
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results
from repcodec.repcodec.RepCodec import RepCodec
from TTS.config import load_config
from TTS.tts.layers.vits.naturalspeech2_pytorch import compute_pitch_pyworld
import utils
from utils import f0_to_coarse,compute_uv,energy_to_coarse
##############################
# IO / Feature extraction
##############################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}


@torch.no_grad()
def weights_reset(m: nn.Module):
    # check if the current module has reset_parameters and if it is reset the weight
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_module_weights_sum(mdl: nn.Module):
    dict_sums = {}
    for name, w in mdl.named_parameters():
        if "weight" in name:
            value = w.data.sum().item()
            dict_sums[name] = value
    return dict_sums


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (0, int((n_fft - hop_length))),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,

        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (0, int(n_fft - hop_length)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec


#############################
# CONFIGS
#############################


@dataclass
class VitsAudioConfig(Coqpit):
    fft_size: int = 1024
    sample_rate: int = 16000
    win_length: int = 1024
    hop_length: int = 320
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = None


##############################
# DATASET
##############################


def get_attribute_balancer_weights(items: list, attr_name: str, multi_dict: dict = None):
    """Create inverse frequency weights for balancing the dataset.
    Use `multi_dict` to scale relative weights."""
    attr_names_samples = np.array([item[attr_name] for item in items])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array([len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names])
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    if multi_dict is not None:
        # check if all keys are in the multi_dict
        for k in multi_dict:
            assert k in unique_attr_names, f"{k} not in {unique_attr_names}"
        # scale weights
        multiplier_samples = np.array([multi_dict.get(item[attr_name], 1.0) for item in items])
        dataset_samples_weight *= multiplier_samples
    return (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )


class VitsDataset(TTSDataset):
    def __init__(self, model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.model_args = model_args

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        tar_item=[d for d in self.samples if d["speaker_name"]==item["speaker_name"]]
        tar_item=random.choice(tar_item)
        tar_wav,_=load_audio(tar_item["audio_file"])



        wav_filename = os.path.basename(item["audio_file"])
        wav_filename=wav_filename.split(".")[0]
        hubert_results=os.path.join(item["root_path"],"whisper_results",item["speaker_name"],wav_filename+".whisper.pt")
        c = torch.load(hubert_results,map_location="cpu")

        f0_results=os.path.join(item["root_path"],"f0_results",item["speaker_name"],wav_filename+".f0.npy")

        f0=np.load(f0_results)

        token_ids = self.get_token_ids(idx, item["text"], item["language"])


        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len or wav.shape[1]>self.max_audio_len or len(token_ids) < self.min_text_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)
        assert len(token_ids)<c.shape[0],(hubert_results,len(token_ids),token_ids,c.shape)



        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "tar_wav":tar_wav,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
            "c":c,
            "f0":f0,
        }

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - audio_unique_names: :math:`[B]`
            -c:
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        max_c_len = max([x.size(0) for x in batch["c"]])
        c_lengths = torch.LongTensor(B)
        max_f0_len=max([x.shape[0] for x in batch["f0"]])



        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        tar_wav_lens=[w.shape[1] for w in batch["tar_wav"]]
        tar_wav_lens = torch.LongTensor(tar_wav_lens)
        tar_wav_lens_max = torch.max(tar_wav_lens)
        tar_wav_rel_lens = tar_wav_lens / tar_wav_lens_max

        c_padded = torch.FloatTensor(B, batch["c"][0].shape[1], max_c_len)
        f0_padded=torch.FloatTensor(B,max_f0_len)



        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        tar_wav_padded=torch.FloatTensor(B, 1, tar_wav_lens_max)
        token_padded = token_padded.zero_()
        wav_padded.zero_()
        c_padded.zero_()
        f0_padded.zero_()
        tar_wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

            tar_wav=batch["tar_wav"][i]
            tar_wav_padded[i,:,:tar_wav.size(1)]=torch.FloatTensor(tar_wav)

            c_padded[i,:,:batch["c"][i].shape[0]]=batch["c"][i].transpose(0,1)
            c_lengths[i]=batch["c"][i].shape[0]

            f0_padded[i,:batch["f0"][i].shape[0]]=torch.from_numpy(batch["f0"][i])

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "tar_wav":tar_wav_padded,
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "tar_waveform_rel_lens":tar_wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
            "c":c_padded,
            "c_lengths":c_lengths,
            "f0":f0_padded,

        }


##############################
# MODEL DEFINITION
##############################


@dataclass
class VitsArgs(Coqpit):
    """VITS model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels of the decoder. Defaults to 513.

        spec_segment_size (int):
            Decoder input segment size. Defaults to 32 `(32 * hoplength = waveform length)`.

        hidden_channels (int):
            Number of hidden channels of the model. Defaults to 192.

        hidden_channels_ffn_text_encoder (int):
            Number of hidden channels of the feed-forward layers of the text encoder transformer. Defaults to 256.

        num_heads_text_encoder (int):
            Number of attention heads of the text encoder transformer. Defaults to 2.

        num_layers_text_encoder (int):
            Number of transformer layers in the text encoder. Defaults to 6.

        kernel_size_text_encoder (int):
            Kernel size of the text encoder transformer FFN layers. Defaults to 3.

        dropout_p_text_encoder (float):
            Dropout rate of the text encoder. Defaults to 0.1.

        dropout_p_duration_predictor (float):
            Dropout rate of the duration predictor. Defaults to 0.1.

        kernel_size_posterior_encoder (int):
            Kernel size of the posterior encoder's WaveNet layers. Defaults to 5.

        dilatation_posterior_encoder (int):
            Dilation rate of the posterior encoder's WaveNet layers. Defaults to 1.

        num_layers_posterior_encoder (int):
            Number of posterior encoder's WaveNet layers. Defaults to 16.

        kernel_size_flow (int):
            Kernel size of the Residual Coupling layers of the flow network. Defaults to 5.

        dilatation_flow (int):
            Dilation rate of the Residual Coupling WaveNet layers of the flow network. Defaults to 1.

        num_layers_flow (int):
            Number of Residual Coupling WaveNet layers of the flow network. Defaults to 6.

        resblock_type_decoder (str):
            Type of the residual block in the decoder network. Defaults to "1".

        resblock_kernel_sizes_decoder (List[int]):
            Kernel sizes of the residual blocks in the decoder network. Defaults to `[3, 7, 11]`.

        resblock_dilation_sizes_decoder (List[List[int]]):
            Dilation sizes of the residual blocks in the decoder network. Defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`.

        upsample_rates_decoder (List[int]):
            Upsampling rates for each concecutive upsampling layer in the decoder network. The multiply of these
            values must be equal to the kop length used for computing spectrograms. Defaults to `[8, 8, 2, 2]`.

        upsample_initial_channel_decoder (int):
            Number of hidden channels of the first upsampling convolution layer of the decoder network. Defaults to 512.

        upsample_kernel_sizes_decoder (List[int]):
            Kernel sizes for each upsampling layer of the decoder network. Defaults to `[16, 16, 4, 4]`.

        periods_multi_period_discriminator (List[int]):
            Periods values for Vits Multi-Period Discriminator. Defaults to `[2, 3, 5, 7, 11]`.

        use_sdp (bool):
            Use Stochastic Duration Predictor. Defaults to True.

        noise_scale (float):
            Noise scale used for the sample noise tensor in training. Defaults to 1.0.

        inference_noise_scale (float):
            Noise scale used for the sample noise tensor in inference. Defaults to 0.667.

        length_scale (float):
            Scale factor for the predicted duration values. Smaller values result faster speech. Defaults to 1.

        noise_scale_dp (float):
            Noise scale used by the Stochastic Duration Predictor sample noise in training. Defaults to 1.0.

        inference_noise_scale_dp (float):
            Noise scale for the Stochastic Duration Predictor in inference. Defaults to 0.8.

        max_inference_len (int):
            Maximum inference length to limit the memory use. Defaults to None.

        init_discriminator (bool):
            Initialize the disciminator network if set True. Set False for inference. Defaults to True.

        use_spectral_norm_disriminator (bool):
            Use spectral normalization over weight norm in the discriminator. Defaults to False.

        use_speaker_embedding (bool):
            Enable/Disable speaker embedding for multi-speaker models. Defaults to False.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_file (List[str]):
            List of paths to the files including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

        detach_dp_input (bool):
            Detach duration predictor's input from the network for stopping the gradients. Defaults to True.

        use_language_embedding (bool):
            Enable/Disable language embedding for multilingual models. Defaults to False.

        embedded_language_dim (int):
            Number of language embedding channels. Defaults to 4.

        num_languages (int):
            Number of languages for the language embedding layer. Defaults to 0.

        language_ids_file (str):
            Path to the language mapping file for the Language Manager. Defaults to None.

        use_speaker_encoder_as_loss (bool):
            Enable/Disable Speaker Consistency Loss (SCL). Defaults to False.

        speaker_encoder_config_path (str):
            Path to the file speaker encoder config file, to use for SCL. Defaults to "".

        speaker_encoder_model_path (str):
            Path to the file speaker encoder checkpoint file, to use for SCL. Defaults to "".

        condition_dp_on_speaker (bool):
            Condition the duration predictor on the speaker embedding. Defaults to True.

        freeze_encoder (bool):
            Freeze the encoder weigths during training. Defaults to False.

        freeze_DP (bool):
            Freeze the duration predictor weigths during training. Defaults to False.

        freeze_PE (bool):
            Freeze the posterior encoder weigths during training. Defaults to False.

        freeze_flow_encoder (bool):
            Freeze the flow encoder weigths during training. Defaults to False.

        freeze_waveform_decoder (bool):
            Freeze the waveform decoder weigths during training. Defaults to False.

        encoder_sample_rate (int):
            If not None this sample rate will be used for training the Posterior Encoder,
            flow, text_encoder and duration predictor. The decoder part (vocoder) will be
            trained with the `config.audio.sample_rate`. Defaults to None.

        interpolate_z (bool):
            If `encoder_sample_rate` not None and  this parameter True the nearest interpolation
            will be used to upsampling the latent variable z with the sampling rate `encoder_sample_rate`
            to the `config.audio.sample_rate`. If it is False you will need to add extra
            `upsample_rates_decoder` to match the shape. Defaults to True.

    """

    num_chars: int = 1024#这个鬼地方
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    periods_multi_period_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: List[str] = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    condition_dp_on_speaker: bool = False
    freeze_hubert_encoder:bool=False
    freeze_hubert_pre:bool=False
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False
    freeze_rep:bool=False
    encoder_sample_rate: int = None
    interpolate_z: bool = True
    reinit_DP: bool = False
    reinit_text_encoder: bool = False


class Vits(BaseTTS):
    """VITS TTS model

    Paper::
        https://arxiv.org/pdf/2106.06103.pdf

    Paper Abstract::
        Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel
        sampling have been proposed, but their sample quality does not match that of two-stage TTS systems.
        In this work, we present a parallel endto-end TTS method that generates more natural sounding audio than
        current two-stage models. Our method adopts variational inference augmented with normalizing flows and
        an adversarial training process, which improves the expressive power of generative modeling. We also propose a
        stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the
        uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the
        natural one-to-many relationship in which a text input can be spoken in multiple ways
        with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS)
        on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly
        available TTS systems and achieves a MOS comparable to ground truth.

    Check :class:`TTS.tts.configs.vits_config.VitsConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> from TTS.tts.models.vits import Vits
        >>> config = VitsConfig()
        >>> model = Vits(config)
    """

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)

        self.init_multispeaker(config)
        self.init_multilingual(config)

        self.init_upsampling()

        self.length_scale = self.args.length_scale
        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.inference_noise_scale_dp = self.args.inference_noise_scale_dp
        self.noise_scale_dp = self.args.noise_scale_dp
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size
        self.embedded_speaker_dim=512

        '''

        self.km_hubert = nn.Parameter(torch.zeros((513, 256), requires_grad=False))
        with fsspec.open("km_hubert.pt", "rb") as f:
            self.km_hubert.data = torch.load(f, map_location="cpu")
        '''

        '''
        cp="checkpoints/config_se.json"
        model_path="checkpoints/model_se.pth"
        se = load_config(cp)

        self.se_fineturn=SE_fineturn(
            input_dim=se.model_params["input_dim"],
            proj_dim=se.model_params["proj_dim"],
            log_input=se.model_params.get("log_input", False),
            use_torch_spec=se.model_params.get("use_torch_spec", False),
            audio_config=se.audio)
        #self.se_fineturn.load_checkpoint(se, model_path, cache=True)
        '''

        configuration = {}
        configuration["hidden_dim"] = 512
        configuration["embedding_dim"] = 512
        configuration["head"] = 8
        configuration["n_layer"] = 4
        configuration["vocab_size"] = 520 + 1
        configuration["phoneme_vocab_size"] = 520
        configuration["EOS"] = 519
        configuration["decoder_start_token_id"] = 520
        #self.LM_model = Text2SemanticDecoder(configuration)


        self.hubert_f0_encoder=Hubert_F0_Encoder(
            out_channels=192,
            hidden_channels=192,
            filter_channels=512,
            n_heads=2,
            n_layers=3,
            gin_channels=self.embedded_speaker_dim,
            kernel_size=3,
            p_dropout=0.1
        )
        self.enc_hubert = HubertEncoder(
            out_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=3,
            kernel_size=3,
            p_dropout=0.1
        )
        self.f0_decoder = F0Decoder(
            1,
            192,
            192,
            2,
            2,
            5,
            0.1,
            512
        )
        '''
        self.split=Split()
        
        

        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )
        self.text_decoder = TextDecoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )'''

        self.liner_ctc=nn.Linear(192,150)
        '''
        self.conv_encoder = Conv_encoder(
            self.tokenizer.characters.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )'''


        self.rep=RepCodec(input_channels=192,output_channels=192,encode_channels=192,decode_channels=192,
                         code_dim=192,num_quantizers_sub=self.num_languages)
        bert_config = BertConfig(
            vocab_size=520,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1500,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=513,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None, )
        '''
        self.bert_model = BertModel(bert_config)
        self.bert_post = nn.Linear(512, 513)'''

        self.posterior_encoder = PosteriorEncoder(
            513,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )
        '''
        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )
        
        self.flow=TransformerCouplingBlock(
            self.args.hidden_channels,
            self.args.hidden_channels,
            filter_channels=256,
            n_heads=2,
            n_layers=3,
            kernel_size=5,
            p_dropout=0.1,
            n_flows=4,
            gin_channels=512,
            share_parameter=False,
        )

        '''
        self.duration_predictor = StochasticDurationPredictor(
                self.args.hidden_channels,
                192,
                3,
                self.args.dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim if self.args.condition_dp_on_speaker else 0,
                language_emb_dim=self.embedded_language_dim,
            )

        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )
        self.cross=Decoder(
            hidden_channels=192,
            filter_channels=256,
            n_heads=8,
            n_layers=4,
        )
        self.cross2 = Decoder(
            hidden_channels=192,
            filter_channels=256,
            n_heads=8,
            n_layers=4,
        )
        '''
        self.flow = TransformerCouplingBlock(
            self.args.hidden_channels,
            self.args.hidden_channels,
            filter_channels=256,
            n_heads=2,
            n_layers=3,
            kernel_size=5,
            p_dropout=0.1,
            n_flows=4,
            gin_channels=512,
            share_parameter=False,
        )'''
        '''
        self.flow1 = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
        )'''


        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            #cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
            )

        print(sum(p.numel() for p in self.disc.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.flow.parameters() if p.requires_grad))

        print(sum(p.numel() for p in self.enc_hubert.parameters() if p.requires_grad))

        print(sum(p.numel() for p in self.rep.parameters() if p.requires_grad))

        print(sum(p.numel() for p in self.hubert_f0_encoder.parameters() if p.requires_grad))

        print(sum(p.numel() for p in self.posterior_encoder.parameters() if p.requires_grad))


        print(sum(p.numel() for p in self.waveform_decoder.parameters() if p.requires_grad))




    def kmeans_predict(
            self,
            X,
            cluster_centers,
    ):
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :param cluster_centers: (torch.tensor) cluster centers
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param device: (torch.device) device [default: 'cpu']
        :return: (torch.tensor) cluster ids
        """

        # convert to float
        #X = X.float()
        # transfer to device
        dis = self.pairwise_distance(X, cluster_centers)
        dis = torch.where(torch.isnan(dis), 0, dis)
        choice_cluster = torch.argmax(dis, dim=1)
        return choice_cluster

    def pairwise_distance(self, data1, data2):
        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)

        norm_A = A.norm(dim=-1, keepdim=True) + 1e-5
        norm_B = B.norm(dim=-1, keepdim=True) + 1e-5

        # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
        A_normalized = A / norm_A
        B_normalized = B / norm_B

        cosine = A_normalized * B_normalized

        # return N*N matrix for pairwise distance
        cosine_dis = cosine.sum(dim=-1).squeeze()
        return cosine_dis
    @property
    def device(self):
        return next(self.parameters()).device

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.speaker_encoder_model_path or not self.args.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio.sample_rate != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio.sample_rate,
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        if self.args.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.args.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)

            self.emb_t = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_t.weight)

        else:
            self.embedded_language_dim = 0

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    def on_epoch_start(self, trainer):  # pylint: disable=W0613
        """Freeze layers at the beginning of an epoch"""
        self._freeze_layers()
        # set the device of speaker encoder
        if self.args.use_speaker_encoder_as_loss:
            self.speaker_manager.encoder = self.speaker_manager.encoder.to(self.device)
            for param in self.speaker_manager.encoder.parameters():
                param.requires_grad = False

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Duration Predictor was not reinit check it !")
            print(" > Duration Predictor was reinit.")

        if self.args.reinit_text_encoder:
            before_dict = get_module_weights_sum(self.text_encoder)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.text_encoder.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.text_encoder)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Text Encoder was not reinit check it !")
            print(" > Text Encoder was reinit.")

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid, _ = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.enc_hubert.parameters():
                param.requires_grad = False

            for param in self.rep.parameters():
                param.requires_grad=False
        if self.args.freeze_hubert_encoder:
            for param in self.enc_hubert.parameters():
                param.requires_grad=False
        if self.args.freeze_hubert_pre:
            for param in self.pre_hubert.parameters():
                param.requires_grad=False
        if self.args.freeze_PE:
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_DP:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False
        if self.args.freeze_rep:
            for param in self.rep.parameters():
                param.requires_grad = False

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, durations = None, None, None, None
        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"]
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        if "durations" in aux_input and aux_input["durations"] is not None:
            durations = aux_input["durations"]
        return sid, g, lid, durations

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def forward_mas(self, outputs, ctc, x_src, x, x_mask, y_mask, lang_emb):
        # find the alignment path
        '''
        Args:
            x_src:
            lang_emb:
            outputs:
            ctc:[B, T_dec, C]
            x:B,C,T_enc
            x_mask:B,1,T_enc
            y_mask:B,1,T_dec

        Returns:
        '''

        ctc=ctc.log_softmax(dim=2)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        B, T_dec = ctc.shape[0:2]
        T_enc = x.shape[2]
        with torch.no_grad():
            logp = torch.Tensor(B, T_enc, T_dec).to(attn_mask.device)
            for i in range(B):
                for j in range(T_enc):
                    logp[i, j, :] = ctc[i, :, x_src[i][j]]

            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        #print(attn_durations[:2])
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                # g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def forward_mas_src(self, outputs,attn, x, x_mask, y_mask, g=None, lang_emb=None):
        # find the alignment path

        # duration predictor

        #attn_durations = attn.sum(3)B 1 Te
        attn_durations=attn

        if 1:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn
    def forward_mas_text(self, outputs, z_p, m_p, x, x_mask, y_mask, g=None, lang_emb=None):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        t = m_p.shape[2]
        t1 = z_p.shape[2]
        B = len(m_p)
        with torch.no_grad():
            logp = torch.zeros(B, t, t1, device=m_p.device)
            #logp.fill_(-1000000)
            for i in range(B):
                flatten = m_p[i].transpose(0, 1)  # C,T
                embed = z_p[i]  # C T
                dist = (
                        flatten.pow(2).sum(1, keepdim=True)
                        - 2 * flatten @ embed
                        + embed.pow(2).sum(0, keepdim=True)
                )
                logp[i][:, :] = -dist
                mask=attn_mask.bool()
                logp[~(mask.squeeze(1))]=-1000000
                logp[:,::2,:]=-10000

            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if 0:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["text_loss_duration"] = loss_duration
        return outputs, attn

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        if self.args.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.args.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

    def kmeans(self,
               c: torch.tensor,
               c_lengths: torch.tensor,
               ):
        h_x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        z_p= self.enc_hubert(c, h_x_mask)
        return z_p


    def forward(  # pylint: disable=dangerous-default-value
        self,
        c:torch.tensor,
        c_lengths: torch.tensor,
        f0: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
        """Forward pass of the model.

        Args:

            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths.
            y (torch.tensor): Batch of input spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            -c:B T C
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            -ctc_pre:B yinsu T
        """

        outputs = {}
        sid, g, lid, _ = self._set_cond_input(aux_input)
        # ssl prenet
        h_x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        h_x = self.enc_hubert(c, h_x_mask)  # logits: 32 443 513
        soft = self.liner_ctc(h_x.transpose(1, 2))
        # f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        y1, zq1, vqloss1, perplexity1=self.rep(h_x.detach())
        #下面是原本的
        # posterior encoder
        kv1,kv1_mask=self.posterior_encoder(y,y_lengths)
        '''
        q1=self.rep.get_codebook().detach()#512,192
        q1=q1.transpose(0,1).unsqueeze(0)
        q1=q1.repeat(32,1,1)
        q1_mask=torch.ones(q1.shape[0],1,q1.shape[2])
        kv2=self.cross1(q1,q1_mask,kv1,kv1_mask)
        kv2_mask=q1_mask
        
        z=self.cross2(q,q_mask,kv2,kv2_mask)'''




        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        # norm_lf0 = utils.normalize_f0(lf0, h_x_mask, random_scale=False)
        pred_lf0 = self.f0_decoder(h_x.detach(), h_x_mask, spk_emb=g)

        z_p, _ = self.hubert_f0_encoder(h_x, h_x_mask, f0=f0_to_coarse(f0))
        z=self.cross(z_p,h_x_mask,kv1,kv1_mask)



        # select a random feature segment for the waveform.decoder z_slice[B,C,T] slice_ids{B}
        z_slice, slice_ids = rand_segments(z, c_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice)


        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )
        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)
            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None


        outputs.update(
            {
                "soft":soft,
                "model_outputs": o,

                "waveform_seg": wav_seg,
                "slice_ids": slice_ids,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "pred_lf0": pred_lf0,
                "lf0": lf0,
                "vq_loss":vqloss1,
                "x":h_x.detach(),
                "y":y1,
            }
        )
        return outputs

    def fine_turn(self,
                  c: torch.tensor,
                  c_lengths: torch.tensor,
                  token: torch.tensor,
                  token_len: torch.tensor,
                  aux_input
                  ):  # 训练后半部分
        outputs = {}

        h_x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        sid, g, lid, _ = self._set_cond_input(aux_input)

        h_x = self.enc_hubert(c, h_x_mask)  # logits: 32 443 513
        #soft = self.liner_ctc(h_x.transpose(1, 2))
        zq, indices = self.rep.encode_id(h_x.detach())
        indices=indices.squeeze(0)
        #indices[:,:-1:2]=indices[:,1::2]
        indices_kz, x_ys_lengths, duration, indices_ys=self.quchong(indices,c_lengths)
        x_mask = torch.unsqueeze(sequence_mask(x_ys_lengths, indices_ys.size(1)), 1).to(c.dtype)
        _,dur_x=self.rep.decode(indices_ys)


        lang_emb = self.emb_l(lid).unsqueeze(-1)
        #x=self.text_encoder(indices_ys, x_ys_lengths, lang_emb)
        loss_AR = self.LM_model(token, token_len, indices_ys, x_ys_lengths,lang_emb)


        outputs, attn = self.forward_mas_src(outputs, duration, dur_x, x_mask, h_x_mask, lang_emb=lang_emb)


        last_hidden_state = self.bert_model(input_ids=indices_kz.int(), attention_mask=h_x_mask.squeeze(1), return_dict=False)[
            0]  # B T 512

        bert_result = self.bert_post(last_hidden_state)




        #z=m_p
        outputs.update(
            {
                "bert_result":bert_result,
                "indices":indices,
                "loss_AR": loss_AR,
                "mask":h_x_mask
            }
        )
        return outputs

    def quchong(self, indices, c_lengths):
        ind = torch.clone(indices).long()
        x = torch.zeros_like(indices, device=indices.device, dtype=torch.long)
        # x.fill_(512)
        # x_lengths=torch.zeros(indices.shape[0],dtype=torch.int32,device=indices.device)
        duration = torch.zeros_like(indices, device=indices.device, dtype=torch.long)
        x_ys = torch.zeros_like(indices, device=indices.device, dtype=torch.long)
        # x_ys.fill_(512)
        tmp = torch.zeros_like(indices).bool()
        tmp[:, :-1] = (ind[:, 1:] - ind[:, :-1]).bool()

        for i in range(tmp.shape[1] - 3, 0, -1):
            size = (tmp[:, i + 1] == True) & (tmp[:, i + 2] == True)
            tmp[size, i] = False

        ind[tmp == True] = -1

        last_id = torch.zeros(indices.shape[0], dtype=torch.int32, device=indices.device)
        last_id.fill_(-2)

        # id = [[i, -1] for i in range(len(indices))]

        id = torch.zeros(indices.shape[0], dtype=torch.int32, device=indices.device)
        id.fill_(-1)

        x_i = torch.arange(indices.size(0), device=indices.device, dtype=torch.int32)

        for i in range(ind.shape[1]):
            size0 = ind[:, i] == -1  #
            size1 = ((ind[:, i] != -1) & (ind[:, i] != last_id))
            size2 = ((ind[:, i] != -1) & (ind[:, i] == last_id))

            x[:, i][size0] = x[:, i - 1][size0]

            x[:, i][size1] = ind[:, i][size1]
            x[:, i][size2] = x[:, i - 1][size2]

            id[size1] = id[size1] + 1

            # x_ys[x_i,id][size1]=ind[:, i][size1]

            x_ys[x_i[size1], id[size1]] = ind[:, i][size1]
            '''
            duration[x_i[size0], id[size0]] = duration[id.tolist()][size0] + 1
            duration[x_i[size2], id[size2]] = duration[id.tolist()][size2] + 1
            duration[x_i[size1], id[size1]] = 1'''
            duration[x_i, id] = duration[x_i, id] + 1

            last_id = ind[:, i]
        x_ys_lengths = id

        x_ys = x_ys[:, :x_ys_lengths.max()]
        duration = duration[:, :x_ys_lengths.max()]
        return x, x_ys_lengths, duration.unsqueeze(1).float(), x_ys
    def yasuo(self,h_x,x_mask):
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(x_mask, 2)
        B,_,T=h_x.shape
        with torch.no_grad():
            logp = torch.zeros(B, T, T, device=h_x.device)
            #logp.fill_(-1000000)
            for i in range(B):
                flatten = h_x[i].transpose(0, 1)  # C,T
                embed = h_x[i]  # C T
                dist = (
                        flatten.pow(2).sum(1, keepdim=True)
                        - 2 * flatten @ embed
                        + embed.pow(2).sum(0, keepdim=True)
                )
                tmp=torch.triu(dist,diagonal=1)
                #xtmp[tmp>-0.01]=-1000000
                logp[i][:, :] = tmp
                #mask=attn_mask[i].bool()
                #logp[~(mask.squeeze(1))]=-1000000
            #attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']
            attn=tanxin_yinsu_numpy(logp,attn_mask.squeeze(1)).unsqueeze(1).detach()

        print("大哥，这不对吧")
        print(attn.sum(2))
        print(attn.sum(3))


        return attn





    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    @torch.no_grad()
    def inference_vc(self,
                     c: torch.tensor,
                     c_lengths: torch.tensor,
                     y: torch.tensor,
                     y_lengths: torch.tensor,
                     aux_input,
                     ):

        outputs = {}
        print("哪里出错了")
        print(c.shape)
        print(c_lengths)
        print(y.shape)
        print(y_lengths)

        sid, g, lid, _ = self._set_cond_input(aux_input)
        # ssl prenet
        h_x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        h_x = self.enc_hubert(c, h_x_mask)  # logits: 32 443 513
        soft = self.liner_ctc(h_x.transpose(1, 2))

        pred_lf0 = self.f0_decoder(h_x.detach(), h_x_mask, spk_emb=g)
        f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        z_p, _ = self.hubert_f0_encoder(h_x, h_x_mask, f0=f0_to_coarse(f0))

        # 下面是原本的
        # posterior encoder

        kv, kv_mask = self.posterior_encoder(y, y_lengths)

        z = self.cross(z_p, h_x_mask, kv, kv_mask)


        o = self.waveform_decoder(z)


        '''
        sid, g, lid, durations = self._set_cond_input(aux_input)

        h_x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)

        h_x = self.enc_hubert(c, h_x_mask)  # logits: 32 443 513


        pred_lf0 = self.f0_decoder(h_x.detach(), h_x_mask, spk_emb=g)

        f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        f0[f0 < 0] = 0



        z_p, _ = self.hubert_f0_encoder(h_x, h_x_mask, f0=f0_to_coarse(f0))

        _,indices=self.rep.encode_id(h_x)


        print(indices)

        kv, kv_mask = self.posterior_encoder(y, y_lengths)

        print(kv.shape)
        print(kv_mask.shape)
        print(z_p.shape)
        print(h_x_mask.shape)

        z = self.cross(z_p, h_x_mask, kv, kv_mask)


        o = self.waveform_decoder((z * h_x_mask)[:, :, : self.max_inference_len])
        '''
        outputs = {
            "model_outputs": o,
            #"alignments": attn.squeeze(1),
            #"durations": w_ceil,
            "y_mask": h_x_mask,
        }
        return outputs


    @torch.no_grad()
    def inference(
        self,
        #c: torch.tensor,
        #c_lengths: torch.tensor,
        #token: torch.tensor,
        #token_len: torch.tensor,
        x,
        aux_input
    ):  # pylint: disable=dangerous-default-value
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        sid, g, lid, durations = self._set_cond_input(aux_input)
        token_len = self._set_x_lengths(x, aux_input)
        print("x",x)
        token=x


        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        print(token.shape)
        print(token_len)

        decoder_input_ids = self.LM_model.infer(x=token.int(), x_lens=token_len,lang_emb=lang_emb)
        print("decoder_input_ids",decoder_input_ids.shape)
        print(decoder_input_ids)
        x_mask=torch.ones_like(decoder_input_ids)
        _,y1=self.rep.decode(decoder_input_ids)
        print("y1",y1.shape)


        if durations is None:
            if 1:
                logw = self.duration_predictor(
                    y1,
                    x_mask,
                    g=g if self.args.condition_dp_on_speaker else None,
                    reverse=True,
                    noise_scale=self.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = self.duration_predictor(
                    x, x_mask, g=g if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
                )
            w = torch.exp(logw) * x_mask * self.length_scale
        else:
            assert durations.shape[-1] == x.shape[-1]
            w = durations.unsqueeze(0)

        w_ceil = torch.ceil(w)
        print("对齐：",w_ceil)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]


        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        indices = torch.einsum("kmn, km -> kn", [attn.float(), decoder_input_ids.float()])

        last_hidden_state =self.bert_model(input_ids=indices.int(), attention_mask=y_mask.squeeze(1),
                        return_dict=False)[
            0]  # B T 512

        bert = self.bert_post(last_hidden_state)
        bert = bert.softmax(dim=2)
        bert = torch.argmax(bert, dim=2)
        print("bert", bert)
        _,y = self.rep.decode(bert)

        print("y.shape",y.shape)
        f0=torch.zeros(y.shape[0],y.shape[2],device=y.device)


        z_p, m_p, logs_p, _ = self.hubert_f0_encoder(x=y, x_mask=y_mask,f0=f0_to_coarse(f0))

        z = self.flow(z_p, y_mask, g=g, reverse=True)


        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len])

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": w_ceil,
            "z": z,
            "z_p": z_p,
            "y_mask": y_mask,
        }
        return outputs

    @torch.no_grad()
    def inference_voice_conversion(
            self,
            c: torch.tensor,
            c_lengths: torch.tensor,
            f0: torch.tensor,
            energy: torch.tensor,
            tar_wav: torch.tensor,
            aux_input,
    ):
        """Inference for voice conversion

        Args:
            reference_wav (Tensor): Reference wavform. Tensor of shape [B, T]
            speaker_id (Tensor): speaker_id of the target speaker. Tensor of shape [B]
            d_vector (Tensor): d_vector embedding of target speaker. Tensor of shape `[B, C]`
            reference_speaker_id (Tensor): speaker_id of the reference_wav speaker. Tensor of shape [B]
            reference_d_vector (Tensor): d_vector embedding of the reference_wav speaker. Tensor of shape `[B, C]`
        """
        # compute spectrograms
        sid, g, lid, durations = self._set_cond_input(aux_input)
        z, m_q, logs_q, y_mask = self.posterior_encoder(c, c_lengths,g=g)
        h_x_mask = torch.unsqueeze(sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)

        z, _, _, y_mask = self.upsampling_z(z, y_lengths=c_lengths, y_mask=h_x_mask)
        print("z:", z.shape)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len])
        outputs = {
            "model_outputs": o,
            # "alignments": attn.squeeze(1),
            "y_mask": y_mask,
        }
        return outputs

    def voice_conversion(self, y, y_lengths, speaker_cond_src, speaker_cond_tgt):
        """Forward pass for voice conversion

        TODO: create an end-point for voice conversion

        Args:
            y (Tensor): Reference spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each reference spectrogram. Tensor of shape [B]
            speaker_cond_src (Tensor): Reference speaker ID. Tensor of shape [B,]
            speaker_cond_tgt (Tensor): Target speaker ID. Tensor of shape [B,]
        """
        assert self.num_speakers > 0, "num_speakers have to be larger than 0."
        # speaker embedding
        if self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            g_src = self.emb_g(torch.from_numpy((np.array(speaker_cond_src))).unsqueeze(0)).unsqueeze(-1)
            g_tgt = self.emb_g(torch.from_numpy((np.array(speaker_cond_tgt))).unsqueeze(0)).unsqueeze(-1)
        elif not self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """

        spec_lens = batch["c_lengths"]


        if optimizer_idx == 0:
            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]

            # generator pass

            outputs = self.forward(
                batch["c"],
                spec_lens,
                batch["f0"],
                batch["tar_spec"],
                batch["tar_spec_lens"],
                batch["waveform"],
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(), outputs["waveform_seg"]
            )

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:

            # compute melspec segment
            with autocast(enabled=False):
                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    batch["mel"].float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_len=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                    soft=self.model_outputs_cache["soft"].float(),
                    tokens=batch["tokens"],
                    tokens_len=batch["token_lens"],
                    lf0=self.model_outputs_cache["lf0"].float(),
                    pred_lf0=self.model_outputs_cache["pred_lf0"].float(),
                    vq_loss=self.model_outputs_cache["vq_loss"].float(),
                    x=self.model_outputs_cache["x"].float(),
                    y=self.model_outputs_cache["y"].float(),
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")
    def fineturn_train_step(self,batch: dict, criterion: nn.Module):
        c = batch["c"]
        c_lenghts = batch["c_lengths"]
        token = batch["tokens"]
        token_len = batch["token_lens"]
        language_ids = batch["language_ids"]
        aux_input = {"language_ids": language_ids}
        outputs=self.fine_turn(c,c_lenghts,token,token_len,aux_input)
        with autocast(enabled=False):
            loss_dict = criterion(loss_duration=outputs["loss_duration"],loss_AR=outputs["loss_AR"],
                                  bert_result=outputs["bert_result"],indices=outputs["indices"],mask=outputs["mask"])
        return outputs,loss_dict


    def _log(self, ap, batch, outputs, name_prefix="train"):  # pylint: disable=unused-argument,no-self-use
        '''
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]
        figures = plot_results(y_hat, y, ap, name_prefix)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name_prefix}/audio": sample_voice}


        alignments = outputs[1]["alignments"]
        align_img = alignments[0].data.cpu().numpy().T

        figures.update(
            {
                "alignment": plot_alignment(align_img, output_fig=False),
            }
        )

        return figures, audios'''

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        '''
        figures, audios = self._log(self.ap, batch, outputs, "train")
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)
        '''

    def fineturn_train_log(self,batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int):
        '''
        alignments = outputs["alignments"]
        align_img = alignments[0].data.cpu().numpy().T

        figures={
                "train/alignment": plot_alignment(align_img, output_fig=False)
            }
        logger.train_figures(steps, figures)
        '''


    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)


    @torch.no_grad()
    def fineturn_eval_step(self, batch: dict, criterion: nn.Module):

        return self.fineturn_train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        '''
        figures, audios = self._log(self.ap, batch, outputs, "eval")
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)'''

    def fineturn_eval_log(self,batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int):
        '''
        alignments = outputs["alignments"]
        align_img = alignments[0].data.cpu().numpy().T

        figures = {
            "eval/alignment": plot_alignment(align_img, output_fig=False)
        }
        logger.train_figures(steps, figures)'''


    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name ,root= None, None, None, None,None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
            elif len(sentence_info) == 5:
                text, speaker_name, style_wav, language_name ,src_wav = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
        }

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """

        '''    
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                root=aux_inputs["root"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

        '''
    @torch.no_grad()
    def fineturn_test_run(self,assets):
        '''
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    '''

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        '''
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])
        '''
    def fineturn_test_log(self, outputs: dict, logger: "Logger", assets: dict, steps: int):
        '''
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    '''

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.name_to_id and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.name_to_id[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_unique_names"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.name_to_id and self.args.use_language_embedding:
            language_ids = [self.language_manager.name_to_id[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch["waveform"])
        else:
            wav = batch["waveform"]

        # compute spectrograms
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        batch["tar_spec"]=wav_to_spec(batch["tar_wav"], ac.fft_size, ac.hop_length, ac.win_length, center=False)

        if self.args.encoder_sample_rate:
            # recompute spec with high sampling rate to the loss
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(
                batch["mel"].shape[2] / self.interpolate_factor
            ), f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["tar_spec_lens"]=(batch["tar_spec"].shape[2] * batch["tar_waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        f = batch["f0"].shape[1]
        c = batch["c"].shape[2]
        if f >= c:
            batch["f0"] = batch["f0"][:, :c]
        else:
            batch["f0"] = nn.functional.pad(batch["f0"], [0, c - f])
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
        weights = None
        data_items = dataset.samples
        '''
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                print(f" > Using weighted sampler for attribute '{attr_name}' with alpha '{alpha}'")
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
                )
                weights = weights * alpha
                print(f" > Attribute weights for '{attr_names}' \n | > {attr_weights}")
        '''
        # input_audio_lenghts = [os.path.getsize(x["audio_file"]) for x in data_items]

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None

        # sampler for DDP
        if batch_sampler is None:
            batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            batch_sampler = (
                DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler
            )  # TODO: check batch_sampler with multi-gpu
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = VitsDataset(
                model_args=self.args,
                samples=samples,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)
            if sampler is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,  # shuffle is done in the dataset.
                    collate_fn=dataset.collate_fn,
                    drop_last=False,  # setting this False might cause issues in AMP training.
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                if num_gpus > 1:
                    loader = DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=config.eval_batch_size if is_eval else config.batch_size,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
                else:

                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters

        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if  k.startswith("liner_ctc.") or k.startswith("enc_hubert.") or k.startswith("hubert_f0_encoder.")
                               or k.startswith("posterior_encoder.") or k.startswith("cross.") or k.startswith("waveform_decoder.") or k.startswith("f0_decoder.")
                               or k.startswith("rep.")
                               )
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def fineturn_get_optimizer(self):
        gen_parameters = chain(params for k, params in self.named_parameters() if
                               k.startswith("emb_l.") or k.startswith("text_encoder.") or
                               k.startswith("bert_model.") or k.startswith("LM_model")or
                               k.startswith("duration_predictor.") or k.startswith("bert_post."))
        return get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_fineturn,
                             parameters=gen_parameters)

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def fineturn_get_lr(self):
        return self.config.lr_fineturn


    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def fineturn_get_scheduler(self, optimizer):

        return get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer)


    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config,self.tokenizer.characters.blank_id)]
    def fineturn_get_criterion(self):
        from TTS.tts.layers.losses import FineturnLoss
        return FineturnLoss(self.config,self.tokenizer.characters.blank_id)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, strict=True, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}

        if self.args.encoder_sample_rate is not None and eval:
            # audio resampler is not used in inference time
            self.audio_resampler = None

        # handle fine-tuning from a checkpoint with additional speakers
        if hasattr(self, "emb_g") and state["model"]["emb_g.weight"].shape != self.emb_g.weight.shape:
            num_new_speakers = self.emb_g.weight.shape[0] - state["model"]["emb_g.weight"].shape[0]
            print(f" > Loading checkpoint with {num_new_speakers} additional speakers.")
            emb_g = state["model"]["emb_g.weight"]
            new_row = torch.randn(num_new_speakers, emb_g.shape[1])
            emb_g = torch.cat([emb_g, new_row], axis=0)
            state["model"]["emb_g.weight"] = emb_g
        # load the model weights
        self.load_state_dict(state["model"], strict=False)

        if eval:
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()
        if not config.model_args.encoder_sample_rate:
            assert (
                upsample_rate == config.audio.hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert (
                upsample_rate == effective_hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)

        tokenizer, new_config = TTSTokenizer.init_from_config(config)#定义总字符的地方

        speaker_manager = SpeakerManager.init_from_config(config, samples)#定义说话人
        language_manager = LanguageManager.init_from_config(config)#定义语言，就那两三种

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path, config.model_args.speaker_encoder_config_path
            )

        return Vits(new_config, ap, tokenizer, speaker_manager, language_manager)


##################################
# VITS CHARACTERS
##################################


class VitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: dict = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:

            with open(ipa_characters, 'r') as f:
                ja = json.load(f)
        super().__init__(ja, punctuations, pad, "&", "*", "<BLNK>", is_unique=True, is_sorted=True)

    def _create_vocab(self):
        print(self._pad, self._punctuations, self._characters, self._blank)
        self._vocab = [self._pad] + [self._blank] + [self._eos] +[self._bos] + list(self._punctuations) + self._characters

        print(self._vocab)
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # pylint: disable=unnecessary-comprehension
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
    @staticmethod
    def init_from_config(config: Coqpit):
        print(config.characters)
        if config.characters is not None:
            _pad = config.characters["pad"]
            _punctuations = config.characters["punctuations"]
            _letters = config.characters["characters"]
            _letters_ipa = config.characters["phonemes"]
            return (
                VitsCharacters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
                config,
            )
        characters = VitsCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos="&",
            bos="*",
            blank=self._blank,
            is_unique=True,
            is_sorted=True,
        )
