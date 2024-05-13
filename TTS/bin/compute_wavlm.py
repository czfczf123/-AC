import torch
from wavlm.WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
def compute_wavlm(wav_input_16khz):
    device=wav_input_16khz.device
    wav_input_16khz.to("cpu")

    checkpoint = torch.load('wavlm/path/to/WavLM-Base+.pt',map_location="cpu")
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])


    # extract the representation of last layer
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)


    rep = model.extract_features(wav_input_16khz)[0]
    return rep.to(device)
if __name__ =='__main__':
    a=torch.randn(32,18347)
    print(compute_wavlm(a).shape)
