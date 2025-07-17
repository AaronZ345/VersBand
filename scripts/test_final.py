import os
from pathlib import Path
import csv
import argparse
import glob
import math
import traceback
import sys
import random
import numpy as np
import torch
from omegaconf import OmegaConf
import soundfile
from tqdm import tqdm
from torch.distributed import init_process_group
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import soundfile as sf
import matplotlib.pyplot as plt
from vocoder.hifigan import HifiGAN, CodeUpsampleHifiGan
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.cfm1_audio_sampler import CFMSampler
from ldm.util import instantiate_from_config
from ldm.modules.encoders.caption_generator import CaptionGenerator2
from ldm.data.joinaudiodataset_anylen import pad_or_cut_xd
import random
import soundfile as sf
import numpy as np
import shutil
import pandas as pd
from typing import Any, Dict, List, Optional, Union

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='/root/autodl-tmp/vocal2music/configs/vocal2music_final.yaml'
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='/root/autodl-tmp/vocal2music/logs/2024-09-18T16-21-51_vocal2music_final/checkpoints/last.ckpt'
    )
    parser.add_argument(
        "--manifest_path",  # 训练时候的 tsv，用来直接构造 infer 时的 dataset
        type=str,
        default='/root/autodl-tmp/data/manifests/vocal_to_accomp/train/v2c_0905/total.tsv'
    )
    parser.add_argument(
        "--other_condition",    
        type=str,
        default='/root/autodl-tmp/data/manifests/vocal_to_accomp/train/v2c_0905/midi.npy'
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,  # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default='1-3',  # use '-' to separate, such as '1-2-3-4'
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='test'
    )
    parser.add_argument(
        "--save_plot",
        action='store_true'
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000
    )
    return parser.parse_args()

def save_df_to_tsv(dataframe, path: Union[str, Path]):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def safe_path(path):
    os.makedirs(Path(path).parent, exist_ok=True)
    return path

def load_samples_from_tsv(tsv_path):
    tsv_path = Path(tsv_path)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    if len(samples) == 0:
        print(f"warning: empty manifest: {tsv_path}")
        return []
    return samples

def load_dict_from_tsv(tsv_path, key):
    samples = load_samples_from_tsv(tsv_path)
    samples = {sample[key]: sample for sample in samples}
    return samples

def initialize_model(config, ckpt, device='cpu'):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device, device, model.cond_stage_model.device)
    sampler = CFMSampler(model,num_timesteps=1000)

    return sampler

def get_codec_from_result(vocal_result_dir):
    output_paths = sorted(glob.glob(f"{vocal_result_dir}/*.txt"))
    output_lines = []
    for output_path in output_paths:
        with open(output_path) as f:
            lines = f.readlines()
            output_lines.extend(lines)
    item_names = []
    pred_items = {}
    for line_idx, line in enumerate(output_lines):
        line = line.strip()
        if line[0] != 'D':
            continue
        item_name, lyrics, pred_code = line.split('\t')
        item_name = item_name[2:]

        pred_code = pred_code.split(' ')
        pred_code = pred_code[pred_code.index('<vocal_start>') + 4:]
        pred_code_ = []
        for i in range(0, len(pred_code), 4):
            pred_code_.append(
                [int(pred_code[i][4:]), int(pred_code[i + 1][4:]) - 1024, int(pred_code[i + 2][4:]) - 2048])  # [T, 3]
        pred_code = np.array(pred_code_).transpose()    # [3, T]
        pred_items[item_name] = pred_code

    return pred_items

def save_mel(spec, save_path):
    fig = plt.figure(figsize=(14, 10))
    heatmap = plt.pcolor(spec, vmin=-6, vmax=1.5)
    fig.colorbar(heatmap)
    fig.savefig(save_path, format='png')
    plt.close(fig)

def handle_exception(err, skipped_name=''):
    _, exc_value, exc_tb = sys.exc_info()
    tb = traceback.extract_tb(exc_tb)[-1]
    if skipped_name != '':
        print(f'skip {skipped_name}, {err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
    else:
        print(f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')


class InferDataset(Dataset):
    def __init__(self, manifest_path,other_condition):
        super().__init__()
        # Load all samples from the manifest
        samples = load_samples_from_tsv(manifest_path)

        # Create a dictionary mapping from 'name' to sample data for quick lookup
        self.items_dict = {sample['name']: sample for sample in samples}

        self.caption_generator = CaptionGenerator2()

        self.mel_num = 80
        self.mel_downsample_rate = 2    # temporal downsample rate
        self.min_factor = 4
        self.min_batch_len = 75
        self.pad_value = -5
        self.unit_upsample_rate = 1
        self.unit_frames_multiple = 2 * self.min_factor  # (本身必须是2的倍数，同时要保证mel是4的倍数，所以只能是8的倍数)
        self.unit_pad_value = 1024 * 3

        self.unit_pad_value = self.pad_value  # the last value
        midi_path = other_condition
        beats_path = other_condition.replace('midi','beats')
        print(f'MIDI path: {midi_path}')
        print(f'Beats path: {beats_path}')
        
        self.midi_dict = np.load(midi_path, allow_pickle=True).item()

        self.beats_dict = np.load(beats_path, allow_pickle=True).item()

        # 随机选择项目名称
        self.pred_list = random.sample(list(self.items_dict.keys()), 200)

        self.items = []
        for item_name in self.pred_list:
            if item_name not in self.items_dict:
                continue
            item = self.items_dict[item_name]
            if float(item['duration']) > 20:
                continue
            item['midi'] = self.midi_dict[item_name]
            item['beats'] = self.beats_dict[item_name]
            item['name'] = item_name
            self.items.append(item)

        del self.items_dict
        del self.midi_dict
        del self.beats_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = self.items[index]

        caption = prompt = ""
        captions = data['caption']
        captions = captions.split('<psep>')
        caption = np.random.choice(captions)
        caption = f"Style: {caption} "

        prompt = self.caption_generator.transcribe(
            key=data['key'],
            key_conf=float(data['key_confidence']),
            avg_pitch=float(data['avg_pitch']),
            tempo=float(data['tempo']),
            tempo_conf=float(data['tempo_confidence']),
            emotion=eval(data['emotion']),
            duration=float(data['wav_len'])
        )
        prompt = f"Musical: {prompt}"

        ori_caption=caption
        caption = caption + prompt

        item = {}
        start = -1
        valid_spec = True
        try:
            spec = np.load(data['mel_path'])  # mel spec [80, T]
            org_spec_len = spec_len = spec.shape[1]
            start=0
            
            # print('spec.shape', org_spec_len)
            # 随机选择
            # if spec.shape[1] > org_spec_len:
            #     print('too long', spec.shape[1])
            #     start = np.random.randint(spec.shape[1] - org_spec_len)
            #     spec = spec[:, start: start + org_spec_len]
            #     spec_len = org_spec_len
        except:
            mel_path = data['mel_path']
            print(f'corrupted: {mel_path}')
            spec = np.ones((self.mel_num, self.min_batch_len)).astype(np.float32) * self.pad_value
            org_spec_len = spec_len = self.min_batch_len
            start = 0
            valid_spec = False

        acoustic = np.load(data['vocal_mel_path'])[:20, :] # mel spec [80, T]
        org_spec_len = spec_len = spec.shape[1]
        midi = np.expand_dims(data['midi'], axis=0) # [1, T]

        if abs(math.ceil(acoustic.shape[1] * self.unit_upsample_rate) - org_spec_len) > 5:  # some bad mel could exists
            print(f'corrupted: {data["vocal_mel_path"]}')
            acoustic = np.ones((20, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * self.unit_pad_value
            midi = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * 128
            beats = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * 2
        acoustic_len = acoustic.shape[1]+75
        if math.ceil(acoustic_len * self.unit_upsample_rate) > org_spec_len:
            if start < 0:
                print('name', data['name'])
                print('valid_spec', valid_spec)
                print('start', start)
                print('spec.shape', spec.shape)
                print('acoustic.shape', acoustic.shape)
            assert start >= 0
            start = round(start * self.unit_upsample_rate)
            start = max(min(start, acoustic_len - math.ceil(org_spec_len / self.unit_upsample_rate) - 1), 0)
            acoustic_len = math.ceil(org_spec_len / self.unit_upsample_rate)
            acoustic = acoustic[:, start: start + acoustic_len]
            midi = midi[:, start: start + acoustic_len]
            beats = beats[:, start: start + acoustic_len]

        acoustic_len = int(math.ceil(acoustic_len / self.unit_frames_multiple) * self.unit_frames_multiple)
        acoustic = pad_or_cut_xd(torch.FloatTensor(acoustic), acoustic_len, dim=1, pad_value=self.pad_value)
        midi = pad_or_cut_xd(torch.FloatTensor(midi), acoustic_len, dim=1, pad_value=0)
        beats = pad_or_cut_xd(torch.FloatTensor(beats), acoustic_len, dim=1, pad_value=0)

        spec_len = math.ceil(acoustic_len * self.unit_upsample_rate)
        spec = pad_or_cut_xd(torch.FloatTensor(spec), spec_len, dim=1, pad_value=self.pad_value)
        assert spec.shape[1] == acoustic.shape[1]

        item['acoustic'] = acoustic
        item['image'] = spec
        item['ori_caption'] = ori_caption
        item["caption"] = caption
        item['name'] = data['name']
        item['midi']=midi
        item['beats']=beats
        
        item['audio_path'] = data['audio_path']
        return item

    def collator(self, samples):
        return samples

def normalize_loudness(wav, target_loudness):
    rms = np.sqrt(np.mean(wav ** 2))
    loudness = 20 * np.log10(rms)
    gain = target_loudness - loudness
    normalized_wav = wav * 10 ** (gain / 20)
    return normalized_wav

@torch.no_grad()
def gen_song(rank, args):
    if args.num_gpus > 1:
        init_process_group(backend=args.dist_config['dist_backend'], init_method=args.dist_config['dist_url'],
                           world_size=args.dist_config['world_size'] * args.num_gpus, rank=rank)

    dataset = InferDataset(args.manifest_path,args.other_condition)
    ds_sampler = DistributedSampler(dataset, shuffle=False) if args.num_gpus > 1 else None
    loader = DataLoader(dataset, sampler=ds_sampler, collate_fn=dataset.collator, batch_size=1, num_workers=40, drop_last=False)

    device = torch.device(f"cuda:{int(rank)}")
    sampler = initialize_model(args.config, args.ckpt, device)
    accomp_vocoder = HifiGAN(vocoder_ckpt='/root/autodl-tmp/SingingDictation/checkpoints/240421-hifigan-01', device=device)

    if args.scales != '' or args.scales is not None:
        scales = [float(s) for s in args.scales.split('-')]
    else:
        scales = [args.scale]

    save_dir = args.save_dir
    loader = tqdm(loader) if rank == 0 else loader
    item_idx = 0
    results = []
    csv_data = {}
    csv_data['audio_path']=[]
    csv_data['caption']=[]
    csv_data['name']=[]
    for batch in loader:
        item = batch[0]
        item_name = item['name']

        midi = item['midi'].to(device)
        beats = item['beats'].to(device)
        acoustic= item['acoustic'].to(device)
        caption = item['caption']
        uncond_caption = ""

        # ################## gt codec + accomp prompt -> mel ###################
        cond_gtcodec_accomp_wavs_dict = {}
        for scale in scales:
            latent_length = int(acoustic.shape[1] * dataset.unit_upsample_rate / dataset.mel_downsample_rate)
            start_code = torch.randn(args.n_samples, sampler.model.first_stage_model.embed_dim, latent_length).to(device=device, dtype=torch.float32)
            shape = [sampler.model.first_stage_model.embed_dim, latent_length]
            condition = {
                'caption': [caption] * args.n_samples,
                'acoustic': {'acoustic':torch.stack([acoustic] * args.n_samples),'midi':torch.stack([midi] * args.n_samples).long(),'beats':torch.stack([beats] * args.n_samples).long(),
                             },
                'name': [item_name] * args.n_samples
            }
            c = sampler.model.get_learned_conditioning(condition)
            uc = None
            if args.scale != 1.0:
                uncondition = {
                    'caption': [uncond_caption] * args.n_samples,
                    'acoustic': {'acoustic':torch.stack([acoustic] * args.n_samples),'midi':torch.stack([midi] * args.n_samples).long(),'beats':torch.stack([beats] * args.n_samples).long(),
                                },
                    'name': [item_name] * args.n_samples
                }
                uc = sampler.model.get_learned_conditioning(uncondition)

            samples_ddim, _ = sampler.sample_cfg(S=args.ddim_steps,
                                                cond=c,
                                                batch_size=args.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                x_T=start_code)
            x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
            cond_gtcodec_accomp_wavs = []
            for idx, spec in enumerate(x_samples_ddim):
                wav = accomp_vocoder(spec.transpose(0, 1).cpu())
                cond_gtcodec_accomp_wavs.append((spec.cpu(), wav))
            cond_gtcodec_accomp_wavs_dict[scale] = cond_gtcodec_accomp_wavs

        gt_vocal_path=item['audio_path'].replace('accomp','vocal')
        gt_vocal=sf.read(gt_vocal_path)[0]
        gt_accomp_path=item['audio_path']
        gt_accomp=sf.read(gt_accomp_path)[0]
        
        for scale in scales:
            cond_gtcodec_accomp_dir = os.path.join(save_dir, f'cond_gtcodec_accomp_scale_{scale}')
            cond_gtcodec_accomp_wavs = cond_gtcodec_accomp_wavs_dict[scale]
            for wav_idx, (spec, wav) in enumerate(cond_gtcodec_accomp_wavs):
                min_length = min(wav.shape[0], gt_vocal.shape[0])
                
                cond_gtcodec_accomp_path = os.path.join(cond_gtcodec_accomp_dir, f"{rank}-{item_idx:04d}[{wav_idx}][accomp].wav")
                wav=normalize_loudness(wav, -23)
                wav=wav[:min_length]
                sf.write(safe_path(cond_gtcodec_accomp_path), wav, 24000, subtype='PCM_16')

                # 保存音频路径和 caption 信息
                csv_data['audio_path'].append(cond_gtcodec_accomp_path)  # 音频路径
                csv_data['caption'].append(caption)                       # 对应的 caption
                csv_data['name'].append(item_name)                        # 对应的 item_name
                
                gt_vocal_path = os.path.join(cond_gtcodec_accomp_dir, f"{rank}-{item_idx:04d}[{wav_idx}][gt_vocal].wav")
                gt_vocal = normalize_loudness(gt_vocal, -23)
                gt_vocal=gt_vocal[:min_length]
                song_wav = gt_vocal
                sf.write(safe_path(gt_vocal_path), song_wav, 24000, subtype='PCM_16')
                
                gt_vocal_path = os.path.join(cond_gtcodec_accomp_dir, f"{rank}-{item_idx:04d}[{wav_idx}][song].wav")
                song_wav = wav[:min_length] + gt_vocal[:min_length]
                sf.write(safe_path(gt_vocal_path), song_wav, 24000, subtype='PCM_16')

                gt_vocal_path = os.path.join(cond_gtcodec_accomp_dir, f"{rank}-{item_idx:04d}[{wav_idx}][gt_accomp].wav")
                gt_accomp=normalize_loudness(gt_accomp, -23)
                sf.write(safe_path(gt_vocal_path), gt_accomp, 24000, subtype='PCM_16')

        results.append(item)
        item_idx += 1
    # 保存CSV文件
    csv_save_path = os.path.join(save_dir, 'clap.csv')
    save_df_to_tsv(pd.DataFrame.from_dict(csv_data),  csv_save_path)

    print(f"CSV 文件保存至: {csv_save_path}")

if __name__ == '__main__':
    args = parse_args()
    args.dist_config = {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54189",
        "world_size": 1
    }
    if args.num_gpus > 1:
        mp.spawn(gen_song, nprocs=args.num_gpus, args=(args,))
    else:
        gen_song(0, args=args)
