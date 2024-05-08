# MIA_Skin_Segmentation and Classification
## Segmentation
In this task, we have tried two approaches. We initially tried the pure U-Net method and in test dataset it can achieve DICE with 0.89. Change the weights of class can slightly improve the result to 0.90 to 0.91. We weight the loss by the lession type due to the unbalance of the dataset. Then we also tried a more advanced Transformer based U-Net and for the same test data, it can achieve DICE with 0.91 - 0.93.
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
  
For single image inference: please use function "process_image" in `skin_inference.py` (manually change model path around line 29)

For test dataseet: please change the data_path correctly in `skin_inference.py`
* Test
  
For dataset evaluation: please use 
`python test.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --img_size 224 --batch_size 24 --mode
cross_contextual_attention --spatial_attention 1`

Please change the "saved model" around line 103 and "data_path" at line 90 accordingly with your data path.

### Approach 2 U-Net with weights
See code at jupyter notebook `Segmentation/Skin_segmentation.ipynb` (output included)

We implement this code using the orininal idea of [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) 
#### Pretrained Model
[Get pre-trained model in this link](https://drive.google.com/file/d/14d3liRrutreSOg0VKl25Kzd0yPbsAPcz/view?usp=sharing)
#### Usage
* Train
  
Change the "data_path" to your training and test dataset
* Test

Under title test, replace model by your saved model from previous code or replace it by pretrained model
 
## Classifications
In this task, we handle the classification of symmetry and the diagnosis of skin lesions. The model is based on the reference that:

`Zhang J, Xie Y, Xia Y, Shen C. Attention Residual Learning for Skin Lesion Classification. IEEE Trans Med Imaging. 2019 Sep;38(9):2092-2103. doi: 10.1109/TMI.2019.2893944. Epub 2019 Jan 21. PMID: 30668469.`

We also thanks to Dongliang Ma for his repo, github link at:

`https://github.com/Vipermdl/ARL.git`

#### Prerequirements
You will need to follow the instruction requirement of "ARL" for setting up the system. The training was performed with pytorch 2.1.0 and cuda 12.2. Python version of 3.9 and higher is recommanded.

#### Pretrained Models:
[Get pre-trained model in this link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/mliu90_jh_edu/EtlUch5vEZFPsO5yL9dtJwABMXyAUGGbMk2JpdyTSYhtAQ?e=F424ne):You will need to put the model into folder "./classification/weights/"

#### Usage
* Train
If you would like to train the model, please follow the instructions in the train.ipynb. You may also need to manually change the network output, changing the loss functions etc.
* Inference
`python test.py --path ABSOLUTE_PATH_TO_THE_TEST_DATASET_FOLDER
Then under each folder, you will find a json file name `FOLDER_NAME_label.json`.

#### Short into to methodology
Detailed discussion can be seen on the report. In short, to realize the diagnosis, we downgraded the 3-class classification into two binary classificaiton tasks. The model trained for the first binary classification is then used as the pre-trained weight for the second-round binary classification, as well as the classification of symmetry.
