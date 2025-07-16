# Versatile Framework for Song Generation with Prompt-based Control

#### Yu Zhang, Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Ruiqi Li, Jingyu Lu, Rongjie Huang, Ruiyuan Zhang, Zhiqing Hong, Ziyue Jiang, Zhou Zhao | Zhejiang University

PyTorch implementation of AccompBand of **[VersBand](https://arxiv.org/abs/2504.19062): Versatile Framework for Song Generation with Prompt-based Control**.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.19062)
[![GitHub Stars](https://img.shields.io/github/stars/AaronZ345/VersBand?style=social)](https://github.com/AaronZ345/VersBand)

Visit our [demo page](https://aaronz345.github.io/VersBandDemo/) for audio samples.

## News

## Key Features
- We propose **VersBand**, a multi-task song generation approach for generating high-quality, aligned songs with prompt-based control.
- We design a decoupled model **VocalBand**, which leverages the flow-matching method to generate singing styles, pitches, and melspectrograms, enabling fast and high-quality vocal synthesis with high-level style control.
- We introduce a flow-based transformer model **AccompBand** to generate high-quality, controllable, aligned accompaniments, with the Band-MOE, selecting suitable experts for enhanced quality, alignment, and control.
- Experimental results demonstrate that VersBand achieves superior objective and subjective evaluations compared to baseline models across multiple **song generation** tasks.


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

