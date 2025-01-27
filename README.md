# X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing

## Introduction
We introduce **<span style="color: #FF4500;">X</span><span style="color: #FF6347;">-</span><span style="color: #FF8C00;">F</span><span style="color: #FF2400;">i</span>**, the first foundation model that achieves modality-invariant multimodal human sensing. This model would require **training only once**, allowing **all sensor modalities** that participated in the training process to **be utilized independently or in any combination** for a wide range of potential applications.

We evaluated **<span style="color: #FF4500;">X</span><span style="color: #FF6347;">-</span><span style="color: #FF8C00;">F</span><span style="color: #FF2400;">i</span>** on HPE and HAR tasks in MM-Fi [[1]](https://openreview.net/pdf?id=1uAsASS1th) and XRF55 [[2]](https://dl.acm.org/doi/10.1145/3643543), demonstrating that **<span style="color: #FF4500;">X</span><span style="color: #FF6347;">-</span><span style="color: #FF8C00;">F</span><span style="color: #FF2400;">i</span>** surpasses previous methods by **MPJPE 24.8%** and **PA-MPJPE 21.4%** on the HPE task, and **accuracy 2.8%** on the HAR task.

```
@misc{chen2024xfimodalityinvariantfoundationmodel,
      title={X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing}, 
      author={Xinyan Chen and Jianfei Yang},
      year={2024},
      eprint={2410.10167},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10167}, 
}
```

## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==x.xx.x` and `torchvision==x.xx.x`).
2. `pip install -r requirements.txt`

## Run
### Download Processed Data
- Please download [MM-Fi datatset](https://github.com/ybhbingo/MMFi_dataset) and [XRF55 datatset](https://github.com/aiotgroup/XRF55-repo) from their official websites.
- Oragnize the downloaed dataset in the following structure:
```
X-Fi
├── Data
    ├── MMFi_Dataset
    │   ├── 
    │   ├── 
    ├── XRF55_Dataset
    │   ├── 
    │   ├── 
```