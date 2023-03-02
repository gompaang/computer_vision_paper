# computer_vision_paper

## paper_list
- computer vision (CNN)
  - image classification
  - object detection
  - segmentation
- vision transformer
- self-supervised learning
- weakly supervised learning
- medical AI
  - classification
  - segmentation  
<br/>


## Computer Vision (CNN)

### image classification
|Name|year|paper|summary|code|
|---|---|---|---|---|
|AlexNet (ImageNet Classification with Deep Convolutional Neural Networks)|NeurPS 2012|[paper](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/AlexNet-2012-14b1e41747b340f380d058e0604ea657)|[code](https://github.com/gompaang/cifar10-classification/blob/master/models/alexnet.py)|
|VGGNet (Very Deep Convolutional Networks For Large-Scale Image Recognition)|ICLR 2015|[paper](https://arxiv.org/pdf/1409.1556.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/VGGNet-2015-3c8abd1c60b646a4a173d57f2f3c4f57)|[code](https://github.com/gompaang/cifar10-classification/blob/master/models/vggnet.py)|
|ResNet (Deep Residual Learning for Image Recognition)|CVPR 2015|[paper](https://arxiv.org/pdf/1512.03385.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/ResNet-2015-ab49bb53ac194b6aac7e56fce4499f98)|[code](https://github.com/gompaang/cifar10-classification/blob/master/models/resnet.py)|
|SENet (Squeeze-and-Excitation Networks)|CVPR 2018|[paper](https://arxiv.org/pdf/1709.01507.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/SENet-2018-481b18a669fe4be88e8a2034e37c1999)||

### object detection
|Name|year|paper|summary|code|
|---|---|---|---|---|
|R-CNN (Rich feature hierarchies for accurate object detection and semantiv segmentation)|ILSVRC 2013|[paper](https://arxiv.org/abs/1311.2524)|[notion](https://mirror-dragonfly-2a2.notion.site/R-CNN-2014-967531ae120f41febd53fe331c9dbc61)||
|Fast R-CNN|2015|[paper](https://arxiv.org/pdf/1504.08083.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/Fast-R-CNN-2015-6aaf793a79c645c1a82cbdd18ed61e36)||
|Faster R-CNN (Towards Real-Time Object Detection with Region Proposal Networks)|NIPS 2015|[paper](https://arxiv.org/pdf/1506.01497.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/Faster-R-CNN-2016-59bca3f94a5d4b13a64cc29f50829d3f)||
|YOLO (You Only Look Once: Unified, Real-Time Object Detection)|2016|[paper](https://arxiv.org/pdf/1506.02640.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/YOLO-2016-bbf62be633864d9496302cc39fed227a)||
|SSD (Single Shot MultiBox Detector)|2016|[paper](https://arxiv.org/pdf/1512.02325.pdf)|[notion](https://mirror-dragonfly-2a2.notion.site/SSD-2016-83db924b7b104af5adf125595c816747)||

### segmentation
|Name|year|paper|summary|code|
|---|---|---|---|---|
|FCN (Fully Convolutional Networks for Semantic Segmentation)|CVPR 2015|[paper](https://arxiv.org/abs/1411.4038)|[tistory](https://hey-stranger.tistory.com/266)||
|U-Net (Convolutional Networks for Biomedical Image Segmentation)|MICCAI 2015|[paper](https://arxiv.org/pdf/1505.04597.pdf)|[tistory](https://hey-stranger.tistory.com/245)|[code](https://github.com/gompaang/pytorch-implementation/blob/main/unet.py)|
|SegNet (A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation)|2015|[paper](https://arxiv.org/pdf/1511.00561v3.pdf)|||
|DeepLab v1 (Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs)|ICLR 2015|[paper](https://arxiv.org/pdf/1412.7062v4.pdf)|||
|DeepLab v2 (DeepLab: Semantic Image Segmentation with Deep Convolutioanl Nets, Atrous Convolution, and Fully Connected CRFs)|TPAMI 2017|[paper](https://arxiv.org/pdf/1606.00915.pdf)|||
|DeepLab v3 (Rethinking Atrous Convolution for Semantic Image Segmentation)|2018|[paper](https://arxiv.org/abs/1706.05587)|||
|DeepLab v3+ (Encoder-Decoder with Atrous Separable Convolution for Semantic Image-Segmentation)|ECCV 2018|[paper](https://arxiv.org/pdf/1802.02611v3.pdf)|||
|PSPNet (Pyramid Scene Parsing Network)|CVPR 2017|[paper](https://arxiv.org/abs/1612.01105)|||
<br/>


## Vision Transformer (ViT)
|Name|year|paper|summary|code|
|---|---|---|---|---|
|ViT (An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale)|ICLR 2021|[paper](https://arxiv.org/pdf/2010.11929.pdf)|[tistory](https://hey-stranger.tistory.com/243)||
|Swin Transformer (Hierarchical Vision Transformer using Shifted Windows)|ICCV 2021|[paper](https://arxiv.org/pdf/2103.14030.pdf)||
|MLP-Mixer (An all-MLP Architecture for Vision)|2021|[paper](https://arxiv.org/pdf/2105.01601.pdf)|||
|MaxViT (MaxViT: Multi-Axis Vision Transformer)|2022|[paper](https://arxiv.org/pdf/2204.01697v4.pdf)|||
<br/>


## Self-supervised learning
|Name|year|paper|summary|code|
|---|---|---|---|---|
|SimCLR (A Single Framework for Contrastive Learning of Visual Representations)|ICML 2020|[paper](https://arxiv.org/pdf/2002.05709.pdf)|[tistory](https://hey-stranger.tistory.com/234)||
|MoCo (Momentum Contrast for Unsupervised Visual Representation Learning)|CVPR 2020|[paper](https://arxiv.org/pdf/1911.05722.pdf)|[tistory](https://hey-stranger.tistory.com/235)||
|BYOL (Bootstrap Your Own Latent A New Approach to Self-Supervised Learning)|NeurIPS 2020|[paper](https://arxiv.org/pdf/2006.07733.pdf)|[tistory](https://hey-stranger.tistory.com/236)||
|DINO (Emerging Properties in Self-supervised Vision Transformers)|2021|[paper](https://arxiv.org/pdf/2104.14294v2.pdf)|[tistory](https://hey-stranger.tistory.com/237)||
|SimCLR v2 (Big Self-Supervised Models are Strong Semi-Supervised Learners)|NeurIPS 2020|[paper](https://arxiv.org/pdf/2006.10029.pdf)|[tistory](https://hey-stranger.tistory.com/254)||
|MoCo v2 (Improved Baselines with Momentum Contrastive Learning)|2020|[paper](https://arxiv.org/pdf/2003.04297.pdf)|[tistory](https://hey-stranger.tistory.com/255)||
|MoCo v3 (An Empirical Study of Training Self-Supervised Vision Transformers)|ICCV 2021|[paper](https://arxiv.org/pdf/2104.02057.pdf)|[tistory](https://hey-stranger.tistory.com/256)||
|BEiT (BEiT: BERT Pre-Training of Image Transformers)|ICLR 2022|[paper](https://arxiv.org/pdf/2106.08254v2.pdf)|||
|data2Vec (data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language)||[paper](https://arxiv.org/pdf/2202.03555v3.pdf)|||
<br/>



## Weakly supervised learning
|Name|year|paper|summary|code|
|---|---|---|---|---|
|CAM (Learning Deep Features for Discriminative Localization)|CVPR 2016|[paper](https://arxiv.org/pdf/1512.04150v1.pdf)|[tistory](https://hey-stranger.tistory.com/257)||
|DSRG (Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing)|CVPR 2018|[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf)|[tistory](https://hey-stranger.tistory.com/259)||
|SEAM (Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation)|CVPR 2020|[paper](https://arxiv.org/pdf/2004.04581v1.pdf)|[tistory](https://hey-stranger.tistory.com/260)||
<br/>



## Medical AI

### classification
|Name|year|paper|summary|code|
|---|---|---|---|---|
|MICLe (Big Self-Supervised Models Advance Medical Image Classifications)|ICCV 2021|[paper](https://arxiv.org/pdf/2101.05224.pdf)|[tistory](https://hey-stranger.tistory.com/242)||

### segmentation
|Name|year|paper|summary|code|
|---|---|---|---|---|
|U-Net (Convolutional Networks for Biomedical Image Segmentation)|MICCAI 2015|[paper](https://arxiv.org/pdf/1505.04597.pdf)|[tistory](https://hey-stranger.tistory.com/245)|[code](https://github.com/gompaang/pytorch-implementation/blob/main/unet.py)|
|TransUNet (Transformers Make Strong Encoders for Medical Image Segmentation)|2021|[paper](https://arxiv.org/pdf/2102.04306.pdf)|[tistory](https://hey-stranger.tistory.com/246)||
|UNETR (UNETR: Transformers for 3D Medical Image Segmentation)|2021|[paper](https://arxiv.org/pdf/2103.10504v3.pdf)|[tistory](https://hey-stranger.tistory.com/247)||
|Swin-Unet (Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation)|2021|[paper](https://arxiv.org/pdf/2105.05537v1.pdf)|||
|TransBTS (Multimodal Brain Tumor Segmentation Using Transformer)|2021|[paper](https://arxiv.org/pdf/2103.04430v2.pdf)|||
|Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis|CVPR 2022|[paper](https://arxiv.org/pdf/2111.14791v2.pdf)|||

