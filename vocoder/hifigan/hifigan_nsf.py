import numpy as np
import librosa
import json
import glob
import re
import os

import torch
from vocoder.hifigan.modules.hifigan_nsf import HifiGanGenerator
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams

def denoise(wav, v=0.1):
    spec = librosa.stft(y=wav, n_fft=1024, hop_length=256,
                        win_length=1024, pad_mode='constant')
    spec_m = np.abs(spec)
    spec_m = np.clip(spec_m - v, a_min=0, a_max=None)
    spec_a = np.angle(spec)

    return librosa.istft(spec_m * np.exp(1j * spec_a), hop_length=256,
                         win_length=1024)

def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if '.yaml' in config_path:
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
    elif '.json' in config_path:
        config = json.load(open(config_path, 'r'))
        state = ckpt_dict["generator"]

    model = HifiGanGenerator(config)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device

total_time = 0

class HifiGAN_NSF(torch.nn.Module):
    def __init__(self, vocoder_ckpt, device=None, use_nsf=True):
        super().__init__()
        self.use_nsf = use_nsf
        base_dir = vocoder_ckpt
        config_path = f'{base_dir}/config.yaml'
        if os.path.exists(config_path):
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
            print('| load HifiGAN: ', ckpt)
            self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)
        else:
            config_path = f'{base_dir}/config.json'
            ckpt = f'{base_dir}/generator_v1'
            if os.path.exists(config_path):
                self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)

    def extract_f0_from_mel(self, mel):
        # 提取 f0 的方法
        import librosa
        import numpy as np

        # 假设已知采样率和其他参数
        sr = 48000
        n_fft = 1024
        hop_length = 256
        win_length = 1024
        n_mels = mel.shape[0]

        # 由于直接从 Mel 频谱图中提取 f0 精度不高，这里使用一个简单的方法进行估计
        # 将 Mel 频谱图逆变换回线性频谱
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_basis_inv = np.linalg.pinv(mel_basis)
        linear_spec = np.dot(mel_basis_inv, mel)

        # 获取幅度谱
        S_abs = np.abs(linear_spec)

        # 使用 librosa 的 piptrack 方法估计频率
        frequencies, magnitudes = librosa.piptrack(S=S_abs, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # 初始化 f0 数组
        f0 = np.zeros(frequencies.shape[1])

        # 遍历每一帧，找到最大幅度对应的频率
        for i in range(frequencies.shape[1]):
            index = magnitudes[:, i].argmax()
            f0[i] = frequencies[index, i]

        return f0

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
            f0 = kwargs.get('f0')
            if self.use_nsf:
                if f0 is None:
                    # 从 mel 提取 f0
                    f0 = self.extract_f0_from_mel(mel)
                f0 = torch.FloatTensor(f0[None, :]).to(device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0:
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        return wav_out

    def vocode(self, mel, **kwargs):
        assert len(mel.shape) == 2
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            f0 = kwargs.get('f0')
            if c.shape[1] != 80:
                c = c.transpose(2, 1)
            if self.use_nsf:
                if f0 is None:
                    # 从 mel 提取 f0
                    f0 = self.extract_f0_from_mel(mel)
                    
                f0 = torch.FloatTensor(f0[None, :]).to(device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0:
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        return wav_out
