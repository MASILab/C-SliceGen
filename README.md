# Reducing Positional Variance in Cross-sectional Abdominal CT Slices with Deep Conditional Generative Models

This repository contains the pytorch implementation of our recent paper:

["Reducing Positional Variance in Cross-sectional Abdominal CT Slices with Deep Conditional Generative Models"](https://link.springer.com/content/pdf/10.1007/978-3-031-16449-1_20.pdf)

The code is written by Xin Yu and adapted from [steveli/partial-encoder-decoder](https://github.com/steveli/partial-encoder-decoder)

<img src="https://github.com/MASILab/C-SliceGen/blob/086d74fc743cdd1e6c129826455060cb0cf376fc/images/method.png" width="600px"/>

## Data preparation
The axial slice are downsampled to 256 $\times$ 256 and saved in .png format. Target slice of each subject need to be selected. image pairs (conditonal, target) information should be saved in .csv file as the format in [example csv](data_csv/pair_example.csv)

## Results
<img src="https://github.com/MASILab/C-SliceGen/blob/086d74fc743cdd1e6c129826455060cb0cf376fc/images/qualitative.png" width="600px"/>


## Citation

If you find our work relevant to your research, please cite:

```bibtex
@inproceedings{yu2022reducing,
  title={Reducing Positional Variance in Cross-sectional Abdominal CT Slices with Deep Conditional Generative Models},
  author={Yu, Xin and Yang, Qi and Tang, Yucheng and Gao, Riqiang and Bao, Shunxing and Cai, Leon Y and Lee, Ho Hin and Huo, Yuankai and Moore, Ann Zenobia and Ferrucci, Luigi and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={202--212},
  year={2022},
  organization={Springer}
}
```

