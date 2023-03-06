# Hand Pose Estimation
## Introduction
This is a project we built for the subject CS406 - Image processing and applications (University of Information Technology - VNUHCM). In this project, we tested the Stacked Hourglass Network model (a fairly well-known model used for Human Pose Estimation). In addition, we switched from the usual bottom-up method to the top-down by adding a hand detect module. Here is the architecture model we use:
<p align="center">
  <img src=OurMethod.png/>
</p>
The hand detect module we use the existing model of victordibia (SSD architecture). With the Stacked Hourglass Network, we implemented based on the work of enghock1 and princeton-vl.
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

  1. Put the downloaded FreiHAND dataset in ./data/
  Link: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
  2. Put the downloaded FreiHAND evaluation set in ./data/
  Link: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip
