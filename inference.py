from torch.utils.data import DataLoader

from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.tts.models.vits import Vits
from TTS.config import load_config
from inference.dataset import VitsDataset
from trainer.trainer_utils import get_optimizer, get_scheduler

model_path="YourTTS/best_model.pth"
config_path="YourTTS/config.json"
tts_config = load_config(config_path)
tts_model=Vits.init_from_config(tts_config)
tts_model.load_checkpoint(tts_config, model_path, eval=True)


src=""
tar=""
src_datasets=[""]
samples=
dataset = VitsDataset(samples=samples)
loader = DataLoader(dataset,
                    collate_fn=dataset.collate_fn,
                    pin_memory=False,)

optimizer=get_optimizer(tts_config.optimizer, tts_config.optimizer_params, tts_config.lr_fineturn,
                             parameters=tts_model.enc_hubert)



epoch=100
for i in range(100):

    for cur_step, batch in enumerate(loader):





