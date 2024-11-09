# C/D-ADMM

This repository is the simplest implementation for the paper '[Online Parallel Multi-Task Relationship Learning via Alternating Direction Method of Multipliers](https://www.sciencedirect.com/science/article/abs/pii/S0925231224014735)', accepted by Neurocomputing Journal.

# Guidance

`data/synthetic.npz`: the synthetic dataset used in the paper.

`src/utils.py`: some commonly used functions.

`src/network.py`: generate different topologies with `networkx` package.

`src/gendata.py`: generate the synthetic data from scratch.

`src/single_task.py`: the `ADMM-Single` baseline compared in the paper.

`src/multi_task.py`: the `C-ADMM` algorithm proposed in the paper.

`src/admm.py`: the `D-ADMM` algorithm proposed in the paper.

# Citation

Please cite our paper if this paper/code helps your research.

```
@article{ADMM2024,
title = {Online parallel multi-task relationship learning via alternating direction method of multipliers},
journal = {Neurocomputing},
volume = {611},
pages = {128702},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.128702},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224014735},
author = {Ruiyu Li and Peilin Zhao and Guangxia Li and Zhiqiang Xu and Xuewei Li}
}
```

