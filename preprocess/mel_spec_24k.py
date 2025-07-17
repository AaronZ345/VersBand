import sys
from preprocess.NAT_mel import MelNet
import os
from tqdm import tqdm
from glob import glob
import math
import pandas as pd
import argparse
from argparse import Namespace
import math
import audioread
from tqdm.contrib.concurrent import process_map
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import json
import pyloudnorm as pyln
import warnings
import torch.multiprocessing as mp
from torch.multiprocessing import Value, Lock


"""
24k 版本的，需要和 codec 的 保持倍数一致
"""
class tsv_dataset(Dataset):
    def __init__(self, tsv_path, sr, mode='none', hop_size=None, target_mel_length=None, target_loudness=-14.0, silence_threshold_db=-40.0, max_silence_duration=3.0) -> None:
        super().__init__()
        if os.path.isdir(tsv_path):
            files = glob(os.path.join(tsv_path, '*.tsv'))
            df = pd.concat([pd.read_csv(file, sep='\t') for file in files])
        else:
            df = pd.read_csv(tsv_path, sep='\t')
        self.audio_paths = []
        self.sr = sr
        self.mode = mode
        self.target_mel_length = target_mel_length
        self.target_loudness = target_loudness  # 设置目标响度
        self.meter = pyln.Meter(sr)  # 创建一个响度计用于计算 LUFS
        self.silence_threshold_db = silence_threshold_db  # 静音检测的分贝阈值
        self.max_silence_duration = max_silence_duration  # 最大静音时长 (秒)

        for t in tqdm(df.itertuples()):
            self.audio_paths.append(getattr(t, 'audio_path'))

    def __len__(self):
        return len(self.audio_paths)

    def pad_wav(self, wav):
        wav_length = wav.shape[-1]
        assert wav_length > 100, "wav is too short, %s" % wav_length
        segment_length = (self.target_mel_length + 1) * self.hop_size
        if segment_length is None or wav_length == segment_length:
            return wav
        elif wav_length > segment_length:
            return wav[:, :segment_length]
        elif wav_length < segment_length:
            temp_wav = torch.zeros((1, segment_length), dtype=torch.float32)
            temp_wav[:, :wav_length] = wav
        return temp_wav

    def normalize_loudness(self, wav):
        # 将 wav 转换为 numpy 格式进行响度归一化
        wav_np = wav.squeeze(0).cpu().numpy()

        # 检查音频数据中是否有 NaN 或 Inf 值
        if np.isnan(wav_np).any() or np.isinf(wav_np).any():
            print(f"Invalid audio data encountered before loudness normalization. Skipping this audio.")
            raise ValueError("Invalid audio data detected.")

        # 计算当前音频的整体响度 (LUFS)
        try:
            loudness = self.meter.integrated_loudness(wav_np)
        except ValueError as e:
            print(f"Loudness calculation error: {e}. Skipping this audio.")
            raise ValueError("Loudness calculation failed.")

        # 计算增益
        gain = self.target_loudness - loudness

        # 限制增益范围，避免过大的增益导致错误
        max_gain = 20.0  # 最大允许增益，单位为 dB
        min_gain = -20.0  # 最小允许增益
        if gain > max_gain:
            # print(f"Gain too large ({gain} dB), limiting to {max_gain} dB.")
            gain = max_gain
            return wav,None
        elif gain < min_gain:
            # print(f"Gain too small ({gain} dB), limiting to {min_gain} dB.")
            return wav,None
            gain = min_gain

        # 归一化音频响度
        normalized_wav_np = pyln.normalize.loudness(wav_np, loudness, self.target_loudness)

        # 再次检查归一化后的数据是否包含 NaN 或 Inf
        if np.isnan(normalized_wav_np).any() or np.isinf(normalized_wav_np).any():
            print(f"Invalid data encountered after loudness normalization. Skipping this audio.")
            raise ValueError("Invalid audio data after normalization.")

        # 检查归一化后音频的最大峰值
        peak = np.abs(normalized_wav_np).max()

        # 如果音频的峰值超过 1.0，进行缩放，避免剪切
        if peak > 1.0:
            normalized_wav_np = normalized_wav_np / peak  # 缩放，避免剪切

        return torch.tensor(normalized_wav_np).unsqueeze(0),True  # 转换回 torch 格式

    def detect_silence(self, wav, threshold_db=-40.0, max_silence_duration=3.0):
        # 计算音频的能量水平，判断是否包含长时间静音
        wav_np = wav.squeeze(0).cpu().numpy()
        energy = np.abs(wav_np)
        threshold = 10 ** (threshold_db / 20)  # 分贝阈值转换为线性幅值
        silence_regions = energy < threshold

        # 计算连续静音段的时长
        silence_duration = 0.0
        sample_duration = 1.0 / self.sr
        for is_silence in silence_regions:
            if is_silence:
                silence_duration += sample_duration
                if silence_duration > max_silence_duration:
                    return True  # 过滤掉超过静音阈值的音频
            else:
                silence_duration = 0.0  # 重置静音计数
        return False

    def __getitem__(self, index):
        skip=0
        audio_path = self.audio_paths[index]
        wav, orisr = torchaudio.load(audio_path)
        if wav.shape[0] != 1:  # 如果是立体声，转换为单声道
            wav = wav.mean(0, keepdim=True)
        wav = torchaudio.functional.resample(wav, orig_freq=orisr, new_freq=self.sr)

        # 检查音频时长是否小于1秒
        audio_duration = wav.shape[-1] / self.sr  # 音频长度（秒）
        if audio_duration < 1.0:
            skip = 1  # 标记为跳过
            return audio_path, wav, skip

        # 响度归一化
        wav,skip_f = self.normalize_loudness(wav)
        
        if skip_f is None:
            skip = 1  # 标记为跳过
            return audio_path, wav, skip
            
        if self.mode == 'pad':
            assert self.target_mel_length is not None
            wav = self.pad_wav(wav)
        return audio_path, wav,skip



def process_audio_by_tsv(rank, args, skip_counter, lock):
    if args.num_gpus > 1:
        init_process_group(backend=args.dist_config['dist_backend'], init_method=args.dist_config['dist_url'],
                           world_size=args.dist_config['world_size'] * args.num_gpus, rank=rank)

    sr = args.audio_sample_rate
    dataset = tsv_dataset(args.tsv_path, sr=sr, mode=args.mode, hop_size=args.hop_size,
                          target_mel_length=args.batch_max_length)
    sampler = DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=16, drop_last=False)

    device = torch.device('cuda:{:d}'.format(rank))
    mel_net = MelNet(args.__dict__)
    mel_net.to(device)
    root = args.save_path
    loader = tqdm(loader) if rank == 0 else loader

    for batch in loader:
        audio_paths, wavs, skip_flag = batch
        if skip_flag.item() == 1:
            # 使用锁确保对 skip_counter 的原子操作
            with lock:
                skip_counter.value += 1
            continue
        wavs = wavs.to(device)
        if args.save_resample:
            for audio_path, wav in zip(audio_paths, wavs):
                psplits = audio_path.split('/')
                wav_name = psplits[-1]
                resample_root, resample_name = root + f'_{sr}', wav_name[:-4] + '_audio.npy'
                resample_dir_name = os.path.join(resample_root, *psplits[1:-1])
                resample_path = os.path.join(resample_dir_name, resample_name)
                os.makedirs(resample_dir_name, exist_ok=True)
                np.save(resample_path, wav.cpu().numpy().squeeze(0))

        if args.save_mel:
            mode = args.mode
            batch_max_length = args.batch_max_length

            for audio_path, wav in zip(audio_paths, wavs):
                rel_audio_path = os.path.relpath(audio_path, args.data_root)
                psplits = rel_audio_path.split('/')
                wav_name = psplits[-1]
                mel_root, mel_name = root, wav_name[:-4] + '_mel.npy'
                mel_dir_name = os.path.join(mel_root, f'mel{mode}{sr}', *psplits[:-1])
                mel_path = os.path.join(mel_dir_name, mel_name)
                if not os.path.exists(mel_path):
                    mel_spec = mel_net(wav).cpu().numpy().squeeze(0)  # (mel_bins, mel_len)
                    if mel_spec.shape[1] <= batch_max_length:
                        if mode == 'tile':  # pad is done in dataset as pad wav
                            n_repeat = math.ceil((batch_max_length + 1) / mel_spec.shape[1])
                            mel_spec = np.tile(mel_spec, reps=(1, n_repeat))
                        elif mode == 'none' or mode == 'pad':
                            pass
                        else:
                            raise ValueError(f'mode:{mode} is not supported')
                    mel_spec = mel_spec[:, :batch_max_length]
                    os.makedirs(mel_dir_name, exist_ok=True)
                    np.save(mel_path, mel_spec)

    print(f'Process {rank} finished. Skipped {skip_counter.value} files.')


def split_list(i_list, num):
    each_num = math.ceil(i_list / num)
    result = []
    for i in range(num):
        s = each_num * i
        e = (each_num * (i + 1))
        result.append(i_list[s:e])
    return result


def drop_bad_wav(item):
    index, path = item
    try:
        with audioread.audio_open(path) as f:
            totalsec = f.duration
            if totalsec < 0.1:
                return index  # index
    except:
        print(f"corrupted wav:{path}")
        return index
    return False


def drop_bad_wavs(tsv_path):  # 'audioset.csv'
    df = pd.read_csv(tsv_path, sep='\t')
    item_list = []
    for item in tqdm(df.itertuples(), 'dropping bad wavs'):
        item_list.append((item[0], getattr(item, 'audio_path')))

    r = process_map(drop_bad_wav, item_list, max_workers=16, chunksize=16)
    bad_indices = list(filter(lambda x: x != False, r))

    print(bad_indices)
    with open('bad_wavs.json', 'w') as f:
        x = [item_list[i] for i in bad_indices]
        json.dump(x, f)
    df = df.drop(bad_indices, axis=0)
    df.to_csv(tsv_path, sep='\t', index=False)


def addmel2tsv(save_dir, tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    mels = glob(f'{save_dir}/mel{args.mode}{args.audio_sample_rate}/**/*_mel.npy', recursive=True)
    name2mel, idx2name, idx2mel = {}, {}, {}
    for mel in mels:
        bn = os.path.basename(mel)[:-8]  # remove _mel.npy
        name2mel[bn] = mel
    for t in df.itertuples():
        idx = int(t[0])
        bn = os.path.basename(getattr(t, 'audio_path'))[:-4]
        idx2name[idx] = bn
    for k, v in idx2name.items():
        idx2mel[k] = name2mel[v]
    df['mel_path'] = df.index.map(idx2mel)
    df.to_csv(tsv_path, sep='\t', index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_duration", type=int, default=20)
    return parser.parse_args()


# 在主函数中进行修改，添加共享变量 skip_counter 和锁 lock
if __name__ == '__main__':
    print('This!!!!')
    pargs = parse_args()
    tsv_path = pargs.tsv_path

    num_gpus = pargs.num_gpus
    batch_max_length = int(pargs.max_duration * 75)
    data_root = '/root/autodl-tmp/data'
    save_path = '/root/autodl-tmp/vocal2music/processed'

    args = {
        'audio_sample_rate': 24000,
        'audio_num_mel_bins': 80,
        'fft_size': 1280,
        'win_size': 1280,
        'hop_size': 320,
        'fmin': 0,
        'fmax': 8000,
        'batch_max_length': batch_max_length,
        'tsv_path': tsv_path,
        'num_gpus': num_gpus,
        'mode': 'none',  # pad, none,
        'save_resample': False,
        'save_mel': True,
        'save_path': save_path,
        'data_root': data_root
    }
    os.makedirs(save_path, exist_ok=True)
    print(f'| save mels in {save_path}')
    args = Namespace(**args)

    # 创建共享变量 skip_counter 和锁 lock
    skip_counter = Value('i', 0)  # 用于统计 skip 数量，'i' 表示整数类型
    lock = Lock()

    if args.num_gpus > 1:
        mp.spawn(process_audio_by_tsv, nprocs=args.num_gpus, args=(args, skip_counter, lock))
    else:
        process_audio_by_tsv(0, args=args, skip_counter=skip_counter, lock=lock)

    # 输出最终的 skip 总数
    print(f"Total files skipped: {skip_counter.value}")
    print("Processing finished.")