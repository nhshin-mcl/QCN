# [CVPR 2024] Blind Image Quality Assessment Based On Geometric Order Learning
Official Pytorch Implementation of the CVPR 2024 paper, "Blind Image Quality Assessment Based on Geometric Order Learning."

Paper
-----------------------------------------------------------------------------
A novel approach to blind image quality assessment, called quality comparison network (QCN), is proposed in this paper, which sorts the feature vectors of input images according to their quality scores in an embedding space. QCN employs comparison transformers (CTs) and score pivots, which act as the centroids of feature vectors of similar-quality images. Each CT updates the score pivots and the feature vectors of input images based on their ordered correlation. To this end, we adopt four loss functions. Then, we estimate the quality score of a test image by searching the nearest score pivot to its feature vector in the embedding space. Extensive experiments show that the proposed QCN algorithm yields excellent image quality assessment performances on various datasets. Furthermore, QCN achieves great performances in cross-dataset evaluation, demonstrating its superb generalization capability.

The full paper can be found via the [link]([https://arxiv.org/abs/2203.13122](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_Blind_Image_Quality_Assessment_Based_on_Geometric_Order_Learning_CVPR_2024_paper.html)).

<!--Please cite our paper if you use our code:-->

Dependencies
-----------------------------------------------------------------------------
- Pytorch
- Python 3


Cite
-----------------------------------------------------------------------------
```c
@inproceedings{shin2024blind,
  title={Blind Image Quality Assessment Based on Geometric Order Learning},
  author={Shin, Nyeong-Ho and Lee, Seon-Ho and Kim, Chang-Su},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
```

Other Order Learning Algorithms
-----------------------------------------------------------------------------
* OL (ICLR 2020) [paper](https://openreview.net/pdf?id=HygsuaNFwr) [code](https://github.com/changsukim-ku/order-learning)
* DRC-ORID (ICLR 2021) [paper](https://openreview.net/pdf?id=Yz-XtK5RBxB) [code](https://github.com/seon92/DRC-ORID)
* Chainization (ECCV 2022) [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730199.pdf) [code](https://github.com/seon92/Chainization)
* GOL (NeurIPS 2022) [paper](https://openreview.net/pdf?id=agNTJU1QNw) [code](https://github.com/seon92/GOL)

License
-----------------------------------------------------------------------------
See [MIT License](https://github.com/nhshin-mcl/MWR/blob/main/LICENSE)
