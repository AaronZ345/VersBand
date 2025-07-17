import sys
import numpy as np
import torch
from typing import TypeVar, Optional, Iterator
import logging
import pandas as pd
from ldm.data.joinaudiodataset_anylen import *
import glob
import math
from ldm.modules.encoders.caption_generator import CaptionGenerator2

logger = logging.getLogger(f'main.{__name__}')

sys.path.insert(0, '.')  # nopep8

"""
adapted from joinaudiodataset_struct_sample_anylen.py
"""

class JoinManifestSpecs(torch.utils.data.Dataset):
    def __init__(self, split, main_spec_dir_path, other_spec_dir_path, mel_num=80, mode='pad', spec_crop_len=1248,
                 pad_value=-5, drop=0, max_tokens=80000, other_condition=None, **kwargs):
        super().__init__()
        self.split = split
        self.max_batch_len = spec_crop_len
        self.min_batch_len = 375
        self.min_factor = 4
        self.mel_num = mel_num
        self.drop = drop
        self.pad_value = pad_value
        self.max_tokens = max_tokens
        assert mode in ['pad', 'tile']
        self.collate_mode = mode
        manifest_files = []
        for dir_path in main_spec_dir_path.split(','):
            manifest_files += glob.glob(f'{dir_path}/*.tsv')
        df_list = [pd.read_csv(manifest, sep='\t') for manifest in manifest_files]
        self.df_main = pd.concat(df_list, ignore_index=True)

        # codec dict
        self.unit_upsample_rate = 1
        self.unit_frames_multiple = 2 * self.min_factor # (本身必须是2的倍数，同时要保证mel是4的倍数，所以只能是8的倍数)
        self.unit_pad_value = self.pad_value  # the last value
        print(f'| loading dict: F0 path: {other_condition}')
        midi_path = other_condition.replace('f0','midi')
        beats_path = other_condition.replace('f0','beats')
        print(f'MIDI path: {midi_path}')
        print(f'Beats path: {beats_path}')
        
        self.f0_dict = np.load(other_condition, allow_pickle=True).item()

        self.midi_dict = np.load(midi_path, allow_pickle=True).item()

        self.beats_dict = np.load(beats_path, allow_pickle=True).item()
        self.caption_generator = CaptionGenerator2()

        if split == 'train':
            self.dataset = self.df_main.iloc[300:]
        elif split == 'valid' or split == 'val':
            self.dataset = self.df_main.iloc[:300]
        elif split == 'test':
            self.df_main = self.add_name_num(self.df_main)
            self.dataset = self.df_main
        else:
            raise ValueError(f'Unknown split {split}')
        self.dataset.reset_index(inplace=True)
        print('dataset len:', len(self.dataset), "drop_rate", self.drop)

    def add_name_num(self, df):
        """each file may have different caption, we add num to filename to identify each audio-caption pair"""
        name_count_dict = {}
        change = []
        for t in df.itertuples():
            name = getattr(t, 'name')
            if name in name_count_dict:
                name_count_dict[name] += 1
            else:
                name_count_dict[name] = 0
            change.append((t[0], name_count_dict[name]))
        for t in change:
            df.loc[t[0], 'name'] = str(df.loc[t[0], 'name']) + f'_{t[1]}'
        return df

    def ordered_indices(self):
        index2dur = self.dataset[['duration']].sort_values(by='duration')
        return list(index2dur.index)

    def collater(self, inputs):
        to_dict = {}
        for l in inputs:
            for k, v in l.items():
                if k in to_dict: # image, acoustic, f0, midi, beats, caption, prompt, name
                    to_dict[k].append(v)
                else:
                    to_dict[k] = [v]

        if self.collate_mode == 'pad':
            to_dict['image'] = collate_1d_or_2d(to_dict['image'], pad_idx=self.pad_value, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor)
            to_dict['acoustic'] = collate_1d_or_2d(to_dict['acoustic'], pad_idx=self.pad_value, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor)
            to_dict['f0'] = collate_1d_or_2d(to_dict['f0'], pad_idx=self.pad_value, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor)
            to_dict['midi'] = collate_1d_or_2d(to_dict['midi'], pad_idx=128, min_len=self.min_batch_len,   # B, C, T
                                                max_len=self.max_batch_len, min_factor=self.min_factor).long()
            to_dict['beats'] = collate_1d_or_2d(to_dict['beats'], pad_idx=2, min_len=self.min_batch_len,   # B, C, T   
                                                max_len=self.max_batch_len, min_factor=self.min_factor).long()

        elif self.collate_mode == 'tile':
            to_dict['image'] = collate_1d_or_2d_tile(to_dict['image'], min_len=self.min_batch_len,
                                                     max_len=self.max_batch_len, min_factor=self.min_factor)
        else:
            raise NotImplementedError
        to_dict['caption'] = {

            'caption': to_dict['caption'],
            'acoustic': {'acoustic': to_dict['acoustic'],'f0': to_dict['f0'],'midi': to_dict['midi'],'beats': to_dict['beats']},
            'name': to_dict['name']
        }

        return to_dict

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        data = self.dataset.iloc[idx]
        caption = prompt = ""
        if np.random.uniform(0, 1) > self.drop:
            captions = data['caption']
            captions = captions.split('<psep>')
            caption = np.random.choice(captions)
            caption = f"Style: {caption} "

        if np.random.uniform(0, 1) > self.drop:
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

        caption = caption + prompt

        item = {}
        start = -1
        valid_spec = True
        try:
            spec = np.load(data['mel_path'])  # mel spec [80, T]
            org_spec_len = spec_len = spec.shape[1]
            
            if spec.shape[1] > self.max_batch_len:
                print('too long', spec.shape[1])
                start = np.random.randint(spec.shape[1] - self.max_batch_len)
                spec = spec[:, start: start + self.max_batch_len]
                spec_len = self.max_batch_len
        except:
            mel_path = data['mel_path']
            print(f'corrupted: {mel_path}')
            spec = np.ones((self.mel_num, self.min_batch_len)).astype(np.float32) * self.pad_value
            org_spec_len = spec_len = self.min_batch_len
            start = 0
            valid_spec = False

        acoustic = np.load(data['vocal_mel_path'])[:20, :]  # mel spec [20, T]
        org_spec_len = spec_len = spec.shape[1]
        f0 = np.expand_dims(self.f0_dict[data['name']], axis=0) # [1, T]
        midi = np.expand_dims(self.midi_dict[data['name']], axis=0) # [1, T]
        beats = np.expand_dims(self.beats_dict[data['name']], axis=0) # [1, T]

        if np.random.uniform(0, 1) < self.drop or not valid_spec:
            acoustic = np.ones((20, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * self.pad_value
            f0 = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * self.pad_value
            midi = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * 128
            beats = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * 2

        if abs(math.ceil(acoustic.shape[1] * self.unit_upsample_rate) - org_spec_len) > 5:  # some bad mel could exists
            print(f'corrupted: {data["vocal_mel_path"]}')
            acoustic = np.ones((20, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * self.unit_pad_value
            f0 = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * self.pad_value
            midi = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * 128
            beats = np.ones((1, math.ceil(spec_len / self.unit_upsample_rate))).astype(np.float32) * 2

        acoustic_len = acoustic.shape[1]
        if math.ceil(acoustic_len * self.unit_upsample_rate) > self.max_batch_len:
            if start < 0:
                print('name', data['name'])
                print('valid_spec', valid_spec)
                print('start', start)
                print('spec.shape', spec.shape)
                print('acoustic.shape', acoustic.shape)
            assert start >= 0
            start = round(start * self.unit_upsample_rate)
            start = max(min(start, acoustic_len - math.ceil(self.max_batch_len / self.unit_upsample_rate) - 1), 0)
            acoustic_len = math.ceil(self.max_batch_len / self.unit_upsample_rate)
            acoustic = acoustic[:, start: start + acoustic_len]
            f0 = f0[:, start: start + acoustic_len]
            midi = midi[:, start: start + acoustic_len]
            beats = beats[:, start: start + acoustic_len]

        acoustic_len = int(math.ceil(acoustic_len / self.unit_frames_multiple) * self.unit_frames_multiple)
        acoustic = pad_or_cut_xd(torch.FloatTensor(acoustic), acoustic_len, dim=1, pad_value=self.pad_value)
        f0 = pad_or_cut_xd(torch.FloatTensor(f0), acoustic_len, dim=1, pad_value=self.pad_value)
        midi = pad_or_cut_xd(torch.FloatTensor(midi), acoustic_len, dim=1, pad_value=128)
        beats = pad_or_cut_xd(torch.FloatTensor(beats), acoustic_len, dim=1, pad_value=2)
        spec_len = math.ceil(acoustic_len * self.unit_upsample_rate)
        spec = pad_or_cut_xd(torch.FloatTensor(spec), spec_len, dim=1, pad_value=self.pad_value)
        item['acoustic'] = acoustic
        item['image'] = spec
        item["caption"] = caption
        # item["prompt"] = prompt
        item['name'] = data['name']
        item['f0']=f0
        item['midi']=midi
        item['beats']=beats
        if self.split == 'test':
            item['f_name'] = data['name']
        return item

    def __len__(self):
        return len(self.dataset)


class JoinSpecsTrain(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('train', **specs_dataset_cfg)


class JoinSpecsValidation(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('valid', **specs_dataset_cfg)


class JoinSpecsTest(JoinManifestSpecs):
    def __init__(self, specs_dataset_cfg):
        super().__init__('test', **specs_dataset_cfg)


class DDPIndexBatchSampler(Sampler):    # 让长度相似的音频的indices合到一个batch中以避免过长的pad
    def __init__(self, indices, batch_size, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, max_tokens=80000) -> None:
        
        if num_replicas is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                print("Not in distributed mode")
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                # raise RuntimeError("Requires distributed package to be available")
                rank = 0
            else:
                rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.indices = indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.batches = self.build_batches()
        print(f"rank: {self.rank}, batches_num {len(self.batches)}")
        # If the dataset length is evenly divisible by replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # print(num_replicas, len(self.batches))
        
        if self.drop_last and len(self.batches) % self.num_replicas != 0:
            self.batches = self.batches[:len(self.batches)//self.num_replicas*self.num_replicas]
        if len(self.batches) > self.num_replicas:
            self.batches = self.batches[self.rank::self.num_replicas]
        else: # may happen in sanity checking
            self.batches = [self.batches[0]]
        print(f"after split batches_num {len(self.batches)}")
        self.shuffle = shuffle
        if self.shuffle:
            self.batches = np.random.permutation(self.batches)
        self.seed = seed

    def set_epoch(self,epoch):
        self.epoch = epoch
        if self.shuffle:
            np.random.seed(self.seed+self.epoch)
            self.batches = np.random.permutation(self.batches)

    def build_batches(self):
        batches, batch = [], []
        for index in self.indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

