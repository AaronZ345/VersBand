import torch
from vocoder.hifigan.modules.hifigan import HifiGanGenerator, CodeUpsampleHifiGanGenerator
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

class HifiGAN(torch.nn.Module):
    def __init__(self, vocoder_ckpt, device=None):
        super().__init__()
        # base_dir = hparams['vocoder_ckpt']
        base_dir = vocoder_ckpt     # ckpt dir
        config_path = f'{base_dir}/config.yaml'
        self.config = config = set_hparams(config_path, global_hparams=False, print_hparams=False)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HifiGanGenerator(config)
        load_ckpt(self.model, base_dir, 'model_gen')
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)  # [1, T, C]
            c = c.transpose(2, 1)
            y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out

    def __call__(self, mel):
        return self.spec2wav(mel)

    def vocode(self, mel):
        assert len(mel.shape) == 2
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            # print('mel.shape', c.shape)
            if c.shape[1] != 80:
                c = c.transpose(2, 1)
            # print('c.shape', c.shape)
            y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out

class CodeUpsampleHifiGan(torch.nn.Module):
    def __init__(self, vocoder_ckpt, device=None):
        super(CodeUpsampleHifiGan, self).__init__()
        # base_dir = hparams['vocoder_ckpt']
        base_dir = vocoder_ckpt     # ckpt dir
        config_path = f'{base_dir}/config.yaml'
        self.config = config = set_hparams(config_path, global_hparams=False, print_hparams=False)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CodeUpsampleHifiGanGenerator(config)
        load_ckpt(self.model, base_dir, 'model_gen')
        self.model.to(self.device)
        self.model.eval()

    def spec2wav(self, mel, **kwargs):
        # mel (T, C)
        device = self.device
        with torch.no_grad():
            if not isinstance(mel, torch.Tensor):
                mel = torch.LongTensor(mel)
            c = mel.unsqueeze(0)
            if device != mel.device:
                c = c.to(device)  # [1, T, C]
            c = c.transpose(2, 1)   # [1, C, T]
            y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out

    def __call__(self, mel):
        return self.spec2wav(mel)
