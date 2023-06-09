# Hand Pose Estimation

<p align="center">
  <img src=demo.gif/ width="50%" height="auto">
</p>

## Introduction

This is a project we built for the Hand Pose Estimation problem. In this project, we tested the Stacked Hourglass Network model (a fairly well-known model used for Human Pose Estimation). In addition, we switched from the usual bottom-up method to the top-down by adding a hand detect module. Here is the architecture model we use:

<p align="center">
  <img src=method.png/>
</p>

The hand detect module we use the existing model of [victordibia](https://github.com/victordibia/handtracking) (SSD architecture). With the Stacked Hourglass Network, we implemented based on the work of [enghock1](https://github.com/enghock1/Real-Time-2D-and-3D-Hand-Pose-Estimation) and [princeton-vl](https://github.com/princeton-vl/pytorch_stacked_hourglass).

## Prepare environment

1. python==3.8.16
2. Install pytorch-cuda==11.7 following [official instruction](https://pytorch.org/):

        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
        
3. Install the necessary dependencies by running:

        pip install -r requirements.txt 

## Prepare dataset

Please organizing your datasets for training and testing following this structure: 

```
Main-folder/
│
├── data/ 
│   ├── FreiHAND_pub_v2 - This folder contain data for training model
|   |   ├── ...
|   |
│   └── FreiHAND_pub_v2_eval - public test images
|       ├── ...
|
└── ...
```

1. Put the downloaded [FreiHAND](https://github.com/lmb-freiburg/freihand) dataset in **./data/**

Link: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip

2. Put the downloaded [FreiHAND](https://github.com/lmb-freiburg/freihand) evaluation set in **./data/**

Link: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip

## Running the code

### Training
In this project, we focus on training Stacked Hourglass Network. As for the hand detect module, we reuse the victordibia's pretrained_model (SSD) without further modified. Train the hourglass network:

    python 1.train.py --config-file "configs/train_FreiHAND_dataset.yaml"
    
The trained model weights (net_hm.pth) will be located at **Main-folder/**. Simply copy and paste the trained model into **./model/trained_models** before evaluate.

### Evaluation

Evaluate on FreiHAND dataset:

    python 2.evaluate_FreiHAND.py --config-file "configs/eval_FreiHAND_dataset.yaml"
    
The visualization results will be saved to **./output/**

### Real-time hand pose estimation

Prepare camera and clear angle, good light, less noisy space. Run the following command line:

    python 3.real_time_2D_hand_pose_estimation.py --config-file "configs/eval_webcam.yaml"
    
_Note: Our model only solves the one-handed recognition problem. If there are 2 or more hands, the model will randomly select one hand to predict. To predict multiple hands, please edit the file 3.real_time_2D_hand_pose_estimation.py (because of resource and time limitations, we don't do this part)._

### Addition

To fine-tune the hyperparameters (BATCH_SIZE, NUM_WORKERS, DATA_SIZE, ...), you can edit the .yaml files in the **./configs/** directory.

## Citation

Newell, A., Yang, K. and Deng, J., 2016. [Stacked Hourglass Networks for Human Pose Estimation](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29). In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14 (pp. 483-499). Springer International Publishing.
