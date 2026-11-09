[Back to VersBand](../README.md)

# Train AccompBand

Run commands from the repository root in the environment described in the README. The example GPU IDs refer to existing devices; adjust them for your training setup.

## Data preparation 

1. Crawl websites to build your own song datasets, then annotate them with automatic tools, like [source–accompaniment separation](https://github.com/Anjok07/ultimatevocalremovergui), [MIDI extraction](https://github.com/RickyL-2000/ROSVOT), [beat tracking](https://github.com/mjhydri/BeatNet), and [music caption annotation](https://github.com/seungheondoh/lp-music-caps).
2. Prepare TSV files that include at least an item_name column, and adapt preprocess/preprocess.py to parse your custom file format accordingly.
3. Preprocess the dataset:
```bash
export PYTHONPATH=.
python preprocess/preprocess.py
```

4. Compute mel-spectrograms:

```bash
python preprocess/mel_spec_24k.py --tsv_path ./data/music24k/music.tsv --num_gpus 4 --max_duration 20
```

5. Post-process:

```bash
python preprocess/postprocess.py
```

6. Download [HIFI-GAN](https://drive.google.com/drive/folders/19DHgcdDHl0WOLulTtpSHPg9h7B7m-b_B?usp=drive_link) as the vocoder in `useful_ckpts/hifigan` and [FLAN-T5](https://huggingface.co/google/flan-t5-large) in `useful_ckpts/flan-t5-large`.

## Training AccompBand

1. Train the VAE module and duration predictor
```bash
python main.py --base configs/ae_accomp.yaml -t --gpus 0,1,2,3,4,5,6,7
```

2. Train the main VersBand model
   
```bash
python main.py --base configs/vocal2music.yaml -t --gpus 0,1,2,3,4,5,6,7
```

*Notes*  
- Adjust the compression ratio in the config files (and related scripts).  
- Change the padding length in the dataloader as needed.
