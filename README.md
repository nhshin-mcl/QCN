# [CVPR 2024] Blind Image Quality Assessment Based On Geometric Order Learning
Official Pytorch Implementation of the CVPR 2024 paper, "Blind Image Quality Assessment Based on Geometric Order Learning."
[Nyeong-Ho Shin](https://scholar.google.com/citations?user=dLCMcXMAAAAJ&hl=en), Seon-Ho Lee, Chang-Su Kim

Paper
-----------------------------------------------------------------------------
A novel approach to blind image quality assessment, called quality comparison network (QCN), is proposed in this paper, which sorts the feature vectors of input images according to their quality scores in an embedding space. QCN employs comparison transformers (CTs) and score pivots, which act as the centroids of feature vectors of similar-quality images. Each CT updates the score pivots and the feature vectors of input images based on their ordered correlation. To this end, we adopt four loss functions. Then, we estimate the quality score of a test image by searching the nearest score pivot to its feature vector in the embedding space. Extensive experiments show that the proposed QCN algorithm yields excellent image quality assessment performances on various datasets. Furthermore, QCN achieves great performances in cross-dataset evaluation, demonstrating its superb generalization capability.

The full paper can be found via the [link](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_Blind_Image_Quality_Assessment_Based_on_Geometric_Order_Learning_CVPR_2024_paper.html).

<!--Please cite our paper if this code helps your work:-->

Dependencies
-----------------------------------------------------------------------------
- Pytorch 2.0.1
- Python 3.8

Datasets
-----------------------------------------------------------------------------
- [KonIQ10k](https://database.mmsp-kn.de/koniq-10k-database.html)
- [SPAQ](https://github.com/h4nwei/SPAQ)
- [BID](https://qualinet.github.io/databases/image/ufrj_blurred_image_database/)
- [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/index.html)
- [FLIVE](https://github.com/baidut/PaQ-2-PiQ)

Train
-----------------------------------------------------------------------------
To train QCN, run the below script. 
```c
python train.py
```
- For other/custom dataset, edit 'config_v1.py' and 'get_df_v1.py'.

Pretrained files
-----------------------------------------------------------------------------
Pretrained files for examples of train and test splits of KonIQ10K and SPAQ are provided at the [link](https://drive.google.com/drive/folders/1F7aBDsAgpNaVcdRhKc_xTY-MMi_RXffw?usp=drive_link).


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

License
-----------------------------------------------------------------------------
See [MIT License](https://github.com/nhshin-mcl/QCN?tab=MIT-1-ov-file#readme)

Acknowledgments
-----------------------------------------------------------------------------
This code is based on the below repos.
* [Geometric order learning](https://github.com/seon92/GOL)
* [kMaX-DeepLab](https://github.com/bytedance/kmax-deeplab)
