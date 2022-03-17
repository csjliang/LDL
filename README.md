# LDL

### [Paper](https://liangjie.xyz/LjHomepageFiles/paper_files/LDL_CVPR2022_paper.pdf) |   [Supplementary Material](https://liangjie.xyz/LjHomepageFiles/paper_files/LDL_CVPR2022_suppl.pdf)

> **Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution** <br>
> Jie Liang\*, Hui Zeng\*, and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> In CVPR 2022.

### Abstract

Single image super-resolution (SISR) with generative adversarial networks (GAN) has recently attracted increasing attention due to its potentials to generate rich details. 
However, the training of GAN is unstable, and it often introduces many perceptually unpleasant artifacts along with the generated details. 
In this paper, we demonstrate that it is possible to train a GAN-based SISR model which can stably generate perceptually realistic details while inhibiting visual artifacts. 
Based on the observation that the local statistics (e.g., residual variance) of artifact areas are often different from the areas of perceptually friendly details, 
we develop a framework to discriminate between GAN-generated artifacts and realistic details, and consequently generate an artifact map to regularize and stabilize the model training process. 
Our proposed locally discriminative learning (LDL) method is simple yet effective, which can be easily plugged in off-the-shelf SISR methods and boost their performance. 
Experiments demonstrate that LDL outperforms the state-of-the-art GAN based SISR methods, 
achieving not only higher reconstruction accuracy but also superior perceptual quality on both synthetic and real-world datasets.


### Citation
If you use this dataset or code for your research, please cite our paper.
```
@inproceedings{jie2022LDL,
  title={Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution},
  author={Liang, Jie and Zeng, Hui and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

### Contact
Should you have any questions, please contact me via `liang27jie@gmail.com`.
