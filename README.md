<h1 align="center"> [🔥 ICLR2025 Spotlight] On Disentangled Training for Nonlinear Transform in Learned Image Compression </h1>

<p align="center">
    <a href="https://arxiv.org/abs/2501.13751"><img src="https://img.shields.io/badge/arXiv-2501.13751-b31b1b.svg" alt="Paper"></a>
    <a href="https://openreview.net/forum?id=U67J0QNtzo"><img src="https://img.shields.io/badge/OpenReview-ICLR'25-blue" alt="Paper"></a>
    <!-- <a href="https://proceedings.mlr.press/v235/hong24c.html"><img src="https://img.shields.io/badge/PRML-ICML'24-122267" alt="Paper"></a> -->
    <a href="https://github.com/qingshi9974/AuxT"><img src="https://img.shields.io/badge/Github-AuxT-brightgreen?logo=github" alt="Github"></a>
    <!-- <a href="https://iclr.cc/media/iclr-2023/Slides/11305.pdf"> <img src="https://img.shields.io/badge/Slides (5 min)-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
    <!-- <a href="https://icml.cc/media/PosterPDFs/ICML%202024/34979.png?t=1721291866.935779"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

by Han Li, Shaohui Li, Wenrui Dai, Maida Cao, Nuowen Kan, Chenglin Li, Junni Zou, and Hongkai Xiong at SJTU, and Tsinghua University

International Conference on Learning Representation (ICLR), 2025.

This repository is the official Pytorch implementation of AuxT. 

## Highlight
The AuxT module is a plug-and-play solution that can accelerate training convergence by 2-3 times for learned image compression, while simultaneously improving rate-distortion performance.

## Dependencies
- python==3.9.19
- torch==2.2.0
- compressai==1.2.4
- einops==0.8.0
- timm==1.0.12
- pywavelets==1.6.0
## Training

``` 
CUDA_VISIBLE_DEVICES=0 python -u train.py --ortho 0.1 --lmbda 0.0483 -d [path of training dataset] --cuda --save_path [path for checkpoint] --save
```

## Evaluation
``` 
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint [path of the pretrained checkpoint] --data [path of testing dataset] --cuda --real
```

## Pretrained Model
The code has been reorganized, and the models have been retrained. Therefore, the results may not be exactly the same as those in the paper.

| Model |Lambda | Metric | ckpt | log | 
|-------|--------|--------|------|----|
|TCM-small+AuxT|0.0483   | MSE   | [ckpt](https://drive.google.com/file/d/1zMK4edRlQ4MsgVGxkfOoIzWRoivi0Qyg/view?usp=sharing)   |[Link](https://drive.google.com/file/d/1NOlxyb_xs6b_rKVDAAiDH2CfJmLBySVS/view?usp=sharing) |
|TCM-small+AuxT |0.0250   | MSE   | [ckpt](https://drive.google.com/file/d/1yTYIwUhgoYIywv1gnbuqQj5Oi0XZP61Y/view?usp=drive_link)  |[Link](https://drive.google.com/file/d/1IQtTwTqRJu8gSkCe77gOMOQXhs_OTkbv/view?usp=sharing) |
|TCM-small+AuxT|0.0130   | MSE   | [ckpt](https://drive.google.com/file/d/1uSnTnNzeAS31istZIdmnCUaJcckf-F7S/view?usp=sharing)  | [Link](https://drive.google.com/file/d/1QUNyxp_AQrw2Wwss8xUYrZU4uQkvgQ89/view?usp=sharing) |
|TCM-small+AuxT|0.0067   | MSE   | [ckpt](https://drive.google.com/file/d/1eVI4nSr1zxStYVfZudJmW8IRIUS--Yfg/view?usp=sharing)   |[Link](https://drive.google.com/file/d/1Xca6OFjfvdZgh2rLMneAbF8T8cZMZ-JA/view?usp=sharing) |
|TCM-small+AuxT|0.0035   | MSE   | [ckpt](https://drive.google.com/file/d/1xGKdLylhvVilFGvMe7vMdIZFpBCt7Cnu/view?usp=sharing)  | [Link](https://drive.google.com/file/d/1Oli33T365SLBlmI5TpM7EftgZXNf3X1y/view?usp=sharing)|
|TCM-small+AuxT|0.0018   | MSE   | [ckpt](https://drive.google.com/file/d/1no_pFVxlMgeR_tD9y75Bqrj5aU4xofGt/view?usp=sharing)  | [Link](https://drive.google.com/file/d/1DAwg_RKpztwqKqN2CQo_wqZy_lzw-ssv/view?usp=sharing)|

| Model |Lambda | Metric | ckpt | log | 
|-------|--------|--------|------|----|
|TCM-small Anchor|0.0483   | MSE   | [ckpt](https://drive.google.com/file/d/1rXSQr-C3hO-7GsU_Ax6xtxrE-DT6jrNb/view?usp=sharing)   |[Link](https://drive.google.com/file/d/1rLHyiT1F_u2UA0aIwP9SpvjKfHnM0Oyn/view?usp=sharing) |
|TCM-small Anchor |0.0250   | MSE   | [ckpt](https://drive.google.com/file/d/10D0Gv1n1BvvEweAQYIJ2Y6aEfgmODCpw/view?usp=sharing)  |[Link](https://drive.google.com/file/d/1czUxhWN1k25JtdNGspmSOE9bKodkkS4a/view?usp=sharing) |
|TCM-small Anchor|0.0130   | MSE   | [ckpt](https://drive.google.com/file/d/1dxSAeiCHuyIo2H3VmLZSL51b7ZlGGw6N/view?usp=sharing)   |[Link](https://drive.google.com/file/d/1RBlNxhW9OIPceW2rtTtTG41COwSrNGlS/view?usp=sharing) |
|TCM-small Anchor|0.0067   | MSE   | [ckpt](https://drive.google.com/file/d/1QScAdXDf7jcpqB2n1WI-pIqru4YbR9U7/view?usp=sharing)   |[Link](https://drive.google.com/file/d/1k7HuiXe7_ZMOAY-n6Q6yR2MLsW2YFmIT/view?usp=sharing) |
|TCM-small Anchor|0.0035   | MSE   | [ckpt](https://drive.google.com/file/d/1ooO4d_nFU4mujaQ17K3ezxfYXLfPhpV0/view?usp=sharing)  | [Link](https://drive.google.com/file/d/1YBbzyPV4qaoX9EDtglBmKjDucGHPXH2T/view?usp=sharing)|
|TCM-small Anchor|0.0018   | MSE   |  [ckpt](https://drive.google.com/file/d/1ycX0O4M3eGKyIWdb9b2UU8MPvpY5kGK_/view?usp=sharing) | [Link](https://drive.google.com/file/d/1mbi1pm54KILy_GiBZGaqi8CJwrZikJng/view?usp=sharing)|


## Citation
```
@article{li2025disentangled,
    title={On disentangled training for nonlinear transform in learned image compression},
    author={Li, Han and Li, Shaohui and Dai, Wenrui and Cao, Maida and Kan, Nuowen and Li, Chenglin and Zou, Junni and Xiong, Hongkai},
    journal={arXiv preprint arXiv:2501.13751},
    year={2025}
}
@inproceedings{
    li2025on,
    title={On Disentangled Training for Nonlinear Transform in Learned Image Compression},
    author={Han Li and Shaohui Li and Wenrui Dai and Maida Cao and Nuowen Kan and Chenglin Li and Junni Zou and Hongkai Xiong},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=U67J0QNtzo}
}
```

## Contact
If you have any problem with this code, please feel free to contact **qingshi9974@sjtu.edu.cn**.
