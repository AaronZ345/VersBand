# Versatile Framework for Song Generation with Prompt-based Control

#### Yu Zhang, Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Ruiqi Li, Jingyu Lu, Rongjie Huang, Ruiyuan Zhang, Zhiqing Hong, Ziyue Jiang, Zhou Zhao | Zhejiang University

PyTorch implementation of AccompBand of **[VersBand (EMNLP 2025)](https://arxiv.org/abs/2504.19062): Versatile Framework for Song Generation with Prompt-based Control**.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.19062)
[![Demo](https://img.shields.io/badge/🚀%20Demo%20Page-blue)](https://aaronz345.github.io/VersBandDemo/) 
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/VersBand?style=social&label=GitHub+Stars)](https://github.com/AaronZ345/VersBand)

Visit our [demo page](https://aaronz345.github.io/VersBandDemo/) for song samples.

## News

- 2025.08: We released the code of AcccompBand!
- 2025.08: VersBand is accepted by EMNLP 2025!

## Key Features
- We propose **VersBand**, a multi-task song generation approach for generating high-quality, aligned songs with prompt-based control.
- We design a decoupled model **VocalBand**, which leverages the flow-matching method to generate singing styles, pitches, and melspectrograms, enabling fast and high-quality vocal synthesis with high-level style control.
- We introduce a flow-based transformer model **AccompBand** to generate high-quality, controllable, aligned accompaniments, with the Band-MOE, selecting suitable experts for enhanced quality, alignment, and control.
- Experimental results demonstrate that VersBand achieves superior objective and subjective evaluations compared to baseline models across multiple **song generation** tasks.

## Quick Start
Since VocalBand is similar to our other SVS models (like [TCSinger](https://github.com/AaronZ345/TCSinger), [TechSinger](https://github.com/gwx314/TechSinger)), we only provide the code of AccompBand in this repo. We give an example of how you can train your own model and infer with AccompBand.

To try on your own song dataset, clone this repo on your local machine with NVIDIA GPU + CUDA cuDNN and follow the instructions below.

### Dependencies

A suitable [conda](https://conda.io/) environment named `versband` can be created
and activated with:

```
conda create -n versband python=3.10
conda install --yes --file requirements.txt
conda activate versband
```

### Multi-GPU

By default, this implementation uses as many GPUs in parallel as returned by `torch.cuda.device_count()`. 
You can specify which GPUs to use by setting the `CUDA_DEVICES_AVAILABLE` environment variable before running the training module.

### Data Preparation 

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
python preprocess/postprocess_data.py
```

6. Download [HIFI-GAN](https://drive.google.com/drive/folders/19DHgcdDHl0WOLulTtpSHPg9h7B7m-b_B?usp=drive_link) as the vocoder in `useful_ckpts/hifigan` and [FLAN-T5](https://huggingface.co/google/flan-t5-large) in `useful_ckpts/flan-t5-large`.

### Training AccompBand

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

### Inference with AccompBand

```bash
python scripts/test_final.py
```

*Replace the checkpoint path and CFG coefficient as required.*


## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[Make-An-Audio-3](https://github.com/Text-to-Audio/Make-An-Audio-3),
[TCSinger2](https://github.com/AaronZ345/TCSinger2)
[Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X)
as described in our code.

## Citations ##

If you find this code useful in your research, please cite our work:
```bib
@article{zhang2025versatile,
  title={Versatile framework for song generation with prompt-based control},
  author={Zhang, Yu and Guo, Wenxiang and Pan, Changhao and Zhu, Zhiyuan and Li, Ruiqi and Lu, Jingyu and Huang, Rongjie and Zhang, Ruiyuan and Hong, Zhiqing and Jiang, Ziyue and others},
  journal={arXiv preprint arXiv:2504.19062},
  year={2025}
}
```

## Disclaimer ##

Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's songs without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=AaronZ345/VersBand)

