import glob
import numpy as np
from tqdm import tqdm
import torchaudio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import random
import os
import csv
import ast
import librosa

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

def generate():
    inputs = ['/root/autodl-tmp/data/manifests/msd_prompts/crawl_2_concat_msd_prompts_refined.tsv',
              '/root/autodl-tmp/data/manifests/msd_prompts/crawl_new_concat_msd_prompts_refined.tsv',
              '/root/autodl-tmp/data/manifests/msd_prompts/yt_crawl_msd_prompts_refined.tsv']
    MANIFEST_COLUMNS = ["name", "dataset", "audio_path", "mel_path"]

    def items_generator(input_file):
        with open(input_file, encoding='utf-8') as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            for item in tqdm(reader):
                yield dict(item)

    skip = 0
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    
    count=0
    for input_file in inputs:
        for i, item in enumerate(items_generator(input_file)):
            parts = item['item_name'].split("<sep>")
            if parts[0] == 'yt_crawl':
                parts[0] = 'yt_song_crawler'
            wav_path = f"/root/autodl-tmp/data/{parts[0]}_sp_demix_24k/{parts[1]}/[{parts[3]}]{parts[2]}.accomp.wav"
            if not os.path.exists(wav_path):
                # print(wav_path)
                skip += 1
                continue
            if not os.path.exists(wav_path.replace('accomp','vocal')):
                # print(wav_path)
                skip += 1
                continue
            count+=1
            mel_path = f"/root/autodl-tmp/data/{parts[0]}_sp_demix_24k/{parts[1]}/[{parts[3]}]{parts[2]}.accomp_mel.npy"
            
            dur= librosa.get_duration(filename=wav_path)
            caption= ast.literal_eval(item['caption'])
            caption='<psep>'.join(caption)
            # for t,cap in enumerate(caption):
            manifest["name"].append(str(item['item_name']))
            manifest["dataset"].append(parts[0])
            manifest["audio_path"].append(wav_path)
            manifest["mel_path"].append(mel_path)
            manifest["name"].append(str(item['item_name']+'vocal'))
            manifest["dataset"].append(parts[0])
            manifest["audio_path"].append(wav_path.replace('accomp','vocal'))
            manifest["mel_path"].append(mel_path.replace('accomp','vocal'))
        print(count)

    print(f"skip: {skip}")
    save_df_to_tsv(pd.DataFrame.from_dict(manifest),  f'/root/autodl-tmp/vocal2music/data/music24k/music.tsv')

if __name__ == '__main__':
    generate()
