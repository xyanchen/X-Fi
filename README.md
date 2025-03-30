# X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing

This repository is the official code implementation of the paper [X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing](https://openreview.net/pdf?id=b42wmsdwmB) published on **ICLR 2025**.\
The paper proposes the first foundation model that achieves modality-invariant multimodal human sensing.
![Motivation of X-Fi](figures/concept.gif)

## Authors
- [Xinyan Chen](),
[Jianfei Yang](https://marsyang.site/)
- [MARS Lab](http://marslab.tech/), School of Mechanical and Aerospace Engineering, Nanyang Technological University 

## Citation
```
@inproceedings{chen2024xfi,
    title={X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing}, 
    author={Chen, Xinyan and Yang, Jianfei},
    booktitle = {International Conference on Learning Representations},
    Month = {May},
    year = {2025}
}
```
## Introduction
We introduce **<span style="color: #FF4500;">X</span><span style="color: #FF6347;">-</span><span style="color: #FF8C00;">F</span><span style="color: #FF2400;">i</span>**, the first foundation model that achieves modality-invariant multimodal human sensing. This model would require **training only once**, allowing **all sensor modalities** that participated in the training process to **be utilized independently or in any combination** for a wide range of potential applications.

We evaluated **<span style="color: #FF4500;">X</span><span style="color: #FF6347;">-</span><span style="color: #FF8C00;">F</span><span style="color: #FF2400;">i</span>** on HPE and HAR tasks in MM-Fi [[1]](https://openreview.net/pdf?id=1uAsASS1th) and XRF55 [[2]](https://dl.acm.org/doi/10.1145/3643543), demonstrating that **<span style="color: #FF4500;">X</span><span style="color: #FF6347;">-</span><span style="color: #FF8C00;">F</span><span style="color: #FF2400;">i</span>** surpasses previous methods by **MPJPE 24.8%** and **PA-MPJPE 21.4%** on the HPE task, and **accuracy 2.8%** on the HAR task.

## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==2.1.1` and `torchvision==0.16.1`).
2. `pip install -r requirements.txt`

## Prepare Datasets and PT Model Weights
### Download Processed Data
- Please download [MM-Fi datatset](https://github.com/ybhbingo/MMFi_dataset) and [XRF55 datatset](https://github.com/aiotgroup/XRF55-repo) from their official websites.
- Remember the dataset saving dir for data loading process.
- Suggest to oragnize the downloaed dataset in the following structure:
```
X-Fi
├── Data
    ├── MMFi_Dataset
    ├── XRF55_Dataset
```
### Download Pretrained Modality-Specific Backbones or Pretrained X-Fi Model
- Please download [Modality-Specific Backbones &  Pretrained X-Fi Model](https://drive.google.com/drive/folders/1ShcQqUd5RnqsTBZ3yM97hrooa7p06O2g?usp=sharing) from cloud storage.
#### For pretrained modality-specific backbones
Oragnize the downloaed `.pt` files into modality-corresponded sub-folders within each tasks's `backbones` or `backbone_models` folder.

Take the example of `MMFi_HAR` task, the organized structure will be:
```
X-Fi
├── MMFi_HAR
|   ├── backbones
|   |   ├── depth_benchmark
|   |   |   ├── depth_Resnet18.pt
|   |   ├── lidar_benchmark
|   |   |   ├── lidar_all_random.pt
|   |   ├── mmwave_benchmark
|   |   |   ├── mmwave_all_random_TD.pt
|   |   ├── RGB_benchmark
|   |   |   ├── RGB_Resnet18.pt
```
#### For pretrained X-Fi model
Unzip the downloaded `pre-trained_weights` folder into corresponding task main folder. e.g.
```
X-Fi
├── MMFi_HAR
|   ├── pre-trained_weights
|   |   ├── mmfi_har_checkpoint.pt
```
## Run
### Change Directory
Before run the scripts, `cd` into different task main folder directory. 

Each task main folder is included in **X-FI folder** as follows:
```
X-Fi
├── MMFi_HAR
├── MMFi_HPE
├── XRF55_HAR
```
### X-Fi Model Training
To train X-Fi model with default setting:

Run: `python run.py --dataset [path/to/corresponding/dataset]`

*Example: `<root_path>/X-Fi/MMFi_HAR > python run.py --dataset d:/Data/My_MMFi_Data/MMFi_Dataset`*

### X-Fi Model Validation
To validate the trained X-Fi model performance on **all modality combinations**:

Run: `python validate_all.py --dataset [path/to/corresponding/dataset] --pt_weights [path/to/saved/pretrained/model/weights]`

 *Example: 
 `<root_path>/X-Fi/MMFi_HAR > python validate_all.py --dataset d:/Data/My_MMFi_Data/MMFi_Dataset --pt_weights ./pre-trained_weights/mmfi_har_checkpoint.pt`*