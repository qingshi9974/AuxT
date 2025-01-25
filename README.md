<h1 align="center">On Disentangled Training for Nonlinear Transform in Learned Image Compression </h1>

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


## Dependencies
- python==3.9.19
- PyTorch==2.2.0
- compressai==1.2.4
- einops==0.8.0
- timm==1.0.12
## Training

``` 
CUDA_VISIBLE_DEVICES='0' python -u train.py --ortho 0.1 --lmbda 0.0483 -d [path of training dataset] --cuda --save_path [path for checkpoint] --save
```

## Evaluation
``` 
CUDA_VISIBLE_DEVICES='0' python eval.py --checkpoint [path of the pretrained checkpoint] --data [path of testing dataset] --cuda --real
```

## Pretrained Model


| Lambda | Metric | ckpt | log | 
|--------|--------|------|----|
|0.0483   | MSE   |    | |
|0.0250   | MSE   |   | |
|0.0130   | MSE   |    | |
|0.0067   | MSE   |    | |
|0.0035   | MSE   |  | |
|0.0018   | MSE   |   | |


## Citation
```
@misc{li2025disentangled,
    title={On Disentangled Training for Nonlinear Transform in Learned Image Compression},
    author={Han Li and Shaohui Li and Wenrui Dai and Maida Cao and Nuowen Kan and Chenglin Li and Junni Zou and Hongkai Xiong},
    year={2025},
    eprint={2501.13751},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Contact
If you have any problem with this code, please feel free to contact **qingshi9974@sjtu.edu.cn**.