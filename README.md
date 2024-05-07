# MIA_Skin_Segmentation and Classification
## Segmentation
In this task, we have tried two approaches. We initially tried the pure U-Net method and in test dataset it can achieve DICE with 0.89. Change the weights of class can slightly improve the result to 0.91 to 0.92. We weight the loss by the lession type due to the unbalance of the dataset. Then we also tried a more advanced Transformer based U-Net and for the same test data, it can achieve DICE with 0.94 - 0.95.
### Approach 1 (result submit) Attention Swin U-Net 
See code under folder `Segmentation/AttSwinUNet-compact`

We modify the code from [Attention swin u-net: Cross-contextual attention mechanism for skin lesion segmentation](https://github.com/NITR098/AttSwinUNet)
#### Prerequirements
1. [Get Swin-T model in this link](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"
2. Please use version through python 3.7 to 3.9 and run `pip install -r requirements.txt ` under folder Segmentation/AttSwinUNet-compact (all operations below also in the same folder)
#### Pretrained Model 
[Get pre-trained model in this link](https://drive.google.com/file/d/1TwhAiZyHHk0baNs9kBFfv7JKG6dTnBCa/view?usp=sharing): Put pretrained model into folder "weights/"
#### Usage
* Train
  
`python train.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 150  --img_size 224 --base_lr 0.05 --batch_size 24 --mode cross_contextual_attention --spatial_attention 1`

You need to change the `data_path` in `train.py` manually around line 100
* Inference
  
For single image inference: please use function "process_image" in `skin_inference.py`
* Test
  
For dataset evaluation: please use 
`python test.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --img_size 224 --batch_size 24 --mode
cross_contextual_attention --spatial_attention 1`

Please change the "saved model" at line 103 and "data_path" at 90 accordingly with your data path.

### Approach 2 U-Net with weights
See code at jupyter notebook `Segmentation/Skin_segmentation.ipynb` (output included)

We implement this code using the orininal idea of [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) 
#### Pretrained Model
[Get pre-trained model in this lik](https://drive.google.com/file/d/14d3liRrutreSOg0VKl25Kzd0yPbsAPcz/view?usp=sharing)
#### Usage
* Train
Change the "data_path to" your training and test dataset
* Test
Under title test, replace model by your saved model from previous code or replace it by pretrained model
 
