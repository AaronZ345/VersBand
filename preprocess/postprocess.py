import os
from pathlib import Path
import re
import csv
import sys

from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import wave
import random

def save_df_to_tsv(dataframe, path):
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

def get_wav_num_frames(path, sr=None):
    try:
        with wave.open(path, 'rb') as f:
            sr_ = f.getframerate()
            if sr is None:
                sr = sr_
            return int(f.getnframes() / (sr_ / sr))
    except wave.Error:
        wav_file, sr_ = sf.read(path, dtype='float32')
        if sr is None:
                sr = sr_
        return int(len(wav_file) / (sr_ / sr))
    except:
        wav_file, sr_ = librosa.core.load(path, sr=sr)
        return len(wav_file)


data_root_dir = '/root/autodl-tmp/data/'
np.random.seed(42)

def generate_different_time_mel_path(mel_path):
    # 查找路径中 [xxxx] 的部分并尝试增加或减少 offset
    match = re.search(r'\[(\d{4})\]', mel_path)
    if not match:
        return None
    
    num = int(match.group(1))
    offsets = list(range(-3, 4))  # 创建偏移量列表从 -3 到 3
    offsets.remove(0)  # 移除 0，因为它不会产生新的路径
    
    random.shuffle(offsets)  # 随机打乱偏移量列表

    for offset in offsets:
        new_num = num + offset
        new_mel_path = re.sub(r'\[\d{4}\]', f'[{new_num:04d}]', mel_path)
        
        if os.path.isfile(new_mel_path):
            return new_mel_path
    
    return None  # 如果尝试所有偏移量后仍然找不到合适的路径

def get_random_item_path(items, current_item_name, key, mel_path):
    current_item_base = re.sub(r'\d{4}', '', current_item_name)

    if len(items) < 50:
        return generate_different_time_mel_path(mel_path)

    # 获取 items 的长度
    item_count = 50

    for _ in range(item_count):  # 尝试访问每个元素，但不会无限循环
        random_index = random.randint(0, len(items) - 1)
        item = items[random_index]
        random_item_base = re.sub(r'\d{4}', '', item['name'])

        # 确保不是同一首歌的相邻片段
        if random_item_base != current_item_base:
            mel_path = item[key]
            if os.path.isfile(mel_path):
                return mel_path

    # 如果所有尝试都失败，则返回 None 或 raise AssertionError
    # assert False, f'Cannot find random item path for {current_item_name}'
    return None

"""
要计算 n_frames (24k情况下)
我感觉diffusion在算的时候，直接channel_wise拼接更合理啊，这样可以更好的对齐
"""

if __name__ == '__main__':
    min_wav_len = 1.0
    max_wav_len = 20

    sample_rate = 24000
    hop_size = 320

    exp_name = 'v2c_0905'

    mel_root_dir = '/root/autodl-tmp/vocal2music/processed/melnone24000'
    result_dir = f'/root/autodl-tmp/data/manifests/vocal_to_accomp/train/{exp_name}'
    delimiter = '<sep>'
    print(f'| result: {result_dir}')

    data_paths = {
        'crawl_2': {
            'manifest': '/root/autodl-tmp/data/manifests/crawl_2_manifest.tsv',
            'f0': '/root/autodl-tmp/data/f0/crawl_2_f0_hop320_24k.npy',
            'midi':'/root/autodl-tmp/data/midi/crawl_2/notes_thr_80.npy',
            'prompt': '/root/autodl-tmp/data/manifests/msd_prompts/crawl_2_msd_prompts_refined.tsv',
            'music_feat': '/root/autodl-tmp/data/manifests/text_to_midi/music_feats/crawl_2_music_feat.tsv',
        },
        'crawl_new': {
            'manifest': '/root/autodl-tmp/data/manifests/crawl_new_manifest.tsv',
            'f0':'/root/autodl-tmp/data/f0/crawl_new_f0_hop320_24k.npy',
            'midi':'/root/autodl-tmp/data/midi/crawl_new/notes_thr_80.npy',
            'prompt': '/root/autodl-tmp/data/manifests/msd_prompts/crawl_new_msd_prompts_refined.tsv',
            'music_feat': '/root/autodl-tmp/data/manifests/text_to_midi/music_feats/crawl_new_music_feat.tsv',
        },
        'yt_crawl': {
            'manifest': '/root/autodl-tmp/data/manifests/yt_crawl_manifest.tsv',
            'f0':'/root/autodl-tmp/data/f0/yt_crawl_f0_hop320_24k.npy',
            'midi': '/root/autodl-tmp/data/midi/yt_crawl/notes_thr_80.npy',
            'prompt': '/root/autodl-tmp/data/manifests/msd_prompts/yt_crawl_msd_prompts_refined.tsv',
            'music_feat': '/root/autodl-tmp/data/manifests/text_to_midi/music_feats/yt_crawl_music_feat.tsv',
        },
    }

    beats_dict = np.load('/root/autodl-tmp/data/beats/beats_dict.npy', allow_pickle=True).item()
    long_silence_items_path = '/root/autodl-tmp/data/manifests/fix/long_silence_items.txt'
    too_short_voiced_items_path = '/root/autodl-tmp/data/manifests/fix/too_short_voiced_items.txt'
    unvoiced_items_path = '/root/autodl-tmp/data/manifests/fix/unvoiced_items.txt'
    long_silence_items = load_dict_from_tsv(long_silence_items_path, 'item_name')
    too_short_voiced_items = load_dict_from_tsv(too_short_voiced_items_path, 'item_name')
    unvoiced_items = load_dict_from_tsv(unvoiced_items_path, 'item_name')
    

    new_items = []
    total_f0_dict = {}
    total_midi_dict = {}
    total_beats_dict = {}
    # total_energy_dict = {}
    skip=0
    for ds_name in data_paths:
        manifest_path = data_paths[ds_name]['manifest']
        f0_path = data_paths[ds_name]['f0']
        midi_path = data_paths[ds_name]['midi']
        prompt_path = data_paths[ds_name]['prompt']
        music_feat_path = data_paths[ds_name]['music_feat']

        manifest_items = load_samples_from_tsv(manifest_path)
        music_feat_items = load_dict_from_tsv(music_feat_path, 'item_name')
        f0_dict = np.load(f0_path, allow_pickle=True).item()
        midi_dict = np.load(midi_path, allow_pickle=True).item()
        prompt_dict = None
        if ds_name != 'musiccap':
            prompt_dict = load_dict_from_tsv(prompt_path, 'item_name')

        for item in tqdm(manifest_items, ncols=80, desc=ds_name, total=len(manifest_items)):
            item_name = item['item_name']

            if item_name not in f0_dict:
                skip+=1
                continue
            
            if item_name not in beats_dict:
                skip+=1
                continue
            
            if item_name not in midi_dict:
                skip+=1
                continue

            if item_name in long_silence_items or item_name in too_short_voiced_items or item_name in unvoiced_items:
                skip+=1
                continue

            wav_len = float(item['wav_len'])
            if wav_len < min_wav_len:
                skip+=1
                continue

            new_item = {
                'name': item_name,
                'audio_path': os.path.join(data_root_dir, item['accomp_24k_path'])
            }

            # mel_path
            wav_path = new_item['audio_path']
            mel_path = os.path.join(mel_root_dir, os.path.relpath(wav_path, data_root_dir))
            mel_path = mel_path[:-4] + '_mel.npy'
            if os.path.isfile(mel_path):
                new_item['mel_path'] = mel_path
            else:
                skip+=1
                continue

            # vocal mel path
            vwav_path = wav_path.replace('accomp', 'vocal')
            vmel_path = os.path.join(mel_root_dir, os.path.relpath(vwav_path, data_root_dir))
            vmel_path = vmel_path[:-4] + '_mel.npy'
            if os.path.isfile(vmel_path):
                new_item['vocal_mel_path'] = vmel_path
            else:
                skip+=1
                continue
            
            accomp_mel=np.load(mel_path)
            vocal_mel=np.load(vmel_path)
            
            assert np.any(vocal_mel != accomp_mel)
            assert vocal_mel.shape[1]<=max_wav_len*24000/320 and accomp_mel.shape[1]<=max_wav_len*24000/320, f'{vocal_mel.shape[1]},{accomp_mel.shape[1]}'
            # assert abs(vocal_mel.shape[1] - math.ceil(float(item['wav_len']) * 24000 / 320)) <= 2, f'{n_frames} {math.ceil(float(item["wav_len"]) * 24000 / 320)}'
            
            if accomp_mel.shape[1]!=vocal_mel.shape[1]:
                print(f'| Skip {item_name}: wav for {accomp_mel.shape},{vocal_mel.shape}')
                skip+=1
                continue             

            # prompts
            if ds_name != 'musiccap':
                prompt = prompt_dict[item_name]['caption']
                # deal with bad backslashes
                backslash_idxs = [m.start() for m in re.finditer(r'\\', prompt)]
                if len(backslash_idxs) > 1:
                    delete_idxs = []
                    for i in range(1, len(backslash_idxs)):
                        if backslash_idxs[i] - backslash_idxs[i - 1] == 1:
                            delete_idxs.append(backslash_idxs[i])
                    text_ = list(prompt)
                    for idx in delete_idxs:
                        text_[idx] = ''
                    prompt = ''.join(text_)
                prompts = eval(prompt)
            else:
                prompts = [item['txt']]
            new_item['caption'] = '<psep>'.join(prompts)

            new_item['dataset'] = ds_name
            new_item['duration'] = item['wav_len']
            if float(new_item['duration'])>max_wav_len:
                new_item['duration'] = item['wav_len']=max_wav_len

            if abs(vocal_mel.shape[1]-float(item['wav_len'])*24000/320)>1:
                # print(f'| Skip {item_name}: wav for {float(item["wav_len"])*24000/320},{vocal_mel.shape[1]}')
                total_frames = get_wav_num_frames(wav_path, 24000)
                new_item['duration'] = item['wav_len']=total_frames/24000
                # print(new_item['duration'])
                if abs(total_frames/320 - accomp_mel.shape[1])>5:
                    print(f'| Skip {wav_path}: wav22222 for {total_frames / 320},{vocal_mel.shape}')
                    assert False

            # f0
            n_frames = len(f0_dict[item_name])
            if n_frames>max_wav_len*24000/320:
                n_frames=int(max_wav_len*24000/320)
                f0_dict[item_name]=f0_dict[item_name][:n_frames]
            if abs(n_frames -vocal_mel.shape[1]) > 5:
                skip+=1
                print(f'| Skip {item_name}: size mismatch for f0 {n_frames} {vocal_mel.shape[1]}')
                continue
            f0_dict[item_name]=f0_dict[item_name][:vocal_mel.shape[1]]
            if len(f0_dict[item_name]) < vocal_mel.shape[1]:
                padding = vocal_mel.shape[1] - len(f0_dict[item_name])
                f0_dict[item_name] = np.pad(f0_dict[item_name], (0, padding), 'constant')

            frames_per_second = 24000 / 320
            # 计算每个音符的帧数
            note_frames = [round(float(dur) * frames_per_second) for dur in midi_dict[item_name]['note_durs']]
            
            # 为每个帧分配音高
            frame_pitches = []
            for pitch, frames in zip(midi_dict[item_name]['pitches'], note_frames):
                frame_pitches.extend([pitch] * frames)

            # 转换为 numpy 数组
            frame_pitches = np.array(frame_pitches)

            # midi
            midi_time=np.sum(list(midi_dict[item_name]['note_durs']))
            if midi_time>max_wav_len:
                midi_time=max_wav_len
                frame_pitches=frame_pitches[:int(midi_time*24000/320)]
            if abs(midi_time- float(item['wav_len']))>1:
                print(f'| Skip {item_name}: midi for {midi_time},{item["wav_len"]},{len(f0_dict[item_name])}')
                assert False
            
            n_frames = len(frame_pitches)
            if abs(n_frames -vocal_mel.shape[1]) > 10:
                skip+=1
                print(f'| Skip {item_name}: size mismatch for midi {n_frames} {vocal_mel.shape[1]}')
                continue
            frame_pitches=frame_pitches[:vocal_mel.shape[1]]    
            if len(frame_pitches) < vocal_mel.shape[1]:
                padding = vocal_mel.shape[1] - len(frame_pitches)
                frame_pitches = np.pad(frame_pitches, (0, padding), 'constant')

            for pitch in frame_pitches:
                assert pitch >= 0 and pitch < 128

            # beats
            beats=beats_dict[item_name]
            
            beat_frames = np.zeros(vocal_mel.shape[1])
            for beat in beats:
                frame = int(beat[0] * 24000 / 320)
                if frame < len(beat_frames):
                    beat_frames[frame] = 1
            
            assert len(beat_frames) == vocal_mel.shape[1] == len(f0_dict[item_name]) == len(frame_pitches) ==accomp_mel.shape[1],f'{len(beat_frames)},{vocal_mel.shape[1]},{len(f0_dict[item_name])},{len(frame_pitches)},{accomp_mel.shape[1]}'
            wav_path = new_item['audio_path']

            # music feat
            if item_name in music_feat_items:
                new_item['key'] = music_feat_items[item_name]['key']
                new_item['key_confidence'] = round(float(music_feat_items[item_name]['key_confidence']), 3)
                new_item['avg_pitch'] = round(float(music_feat_items[item_name]['avg_pitch']), 1)
                new_item['tempo'] = round(float(music_feat_items[item_name]['tempo']), 1)
                new_item['tempo_confidence'] = round(float(music_feat_items[item_name]['tempo_confidence']), 3)
                new_item['emotion'] = music_feat_items[item_name]['emotion']
            else:
                new_item['key'] = 'None'
                new_item['key_confidence'] = 0.
                new_item['avg_pitch'] = -1.
                new_item['tempo'] = -1.
                new_item['tempo_confidence'] = 0.
                new_item['emotion'] = 'None'
            wav_len = float(item['wav_len'])
            new_item['wav_len'] = round(wav_len, 4)

            new_items.append(new_item)
            total_f0_dict[item_name] = f0_dict[item_name]
            total_midi_dict[item_name] = frame_pitches
            total_beats_dict[item_name] = beat_frames

    # total
    manifest_columns = list(new_items[0].keys())
    manifest = {c: [] for c in manifest_columns}
    for sample in new_items:
        for c in manifest_columns:
            manifest[c].append(sample[c])
    result_path = os.path.join(result_dir, 'total.tsv')
    os.makedirs(Path(result_path).parent, exist_ok=True)
    save_df_to_tsv(pd.DataFrame.from_dict(manifest), result_path)
    print(f'| Totally {len(new_items)} samples.')
    print(f'| Skip {skip} samples.')
    
    # f0
    np.save(os.path.join(result_dir, 'f0.npy'), total_f0_dict, allow_pickle=True)
    np.save(os.path.join(result_dir, 'midi.npy'), total_midi_dict, allow_pickle=True)
    np.save(os.path.join(result_dir, 'beats.npy'), total_beats_dict, allow_pickle=True)