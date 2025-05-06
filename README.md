# BDAA project: identifying traffic signs with CNN

## Description

This project is a Spring 2025 BDAA project. This project used supervised learning to solve the captioned image recognition problem. A Convolutional neural network (CNN) is trained and evaluated on the dataset. Moreover, the project explores related phenomena such as overfitting and underfitting, as well as the impact of techniques such as Batch Normalization. A visualization of the CNN is also given. 


## Project Structure Overview

```
BDAA-final-project/
├── data/                                       # Data storage
├── results/                                    # results of explorations
├── main.py                                     # the main program for traffic sign identification
├── observe_overfitting.py                      # to observe overfitting when regularization is weak
├── observe_underfitting.py                     # to observe underfitting when regularization is strong
├── data_preprocessing_batchnorm.py             # to study the impact of data preprocessing and batch normalization on test accuracy
├── kernel_visualization.py                     # to visualize the first layer filters
├── utils.py                                    # functions that will be frequently used, including those for loading data, preprocessing data,and providing the Convolutional Neural Network Model
├── requirements.txt                            # required packages
└── README.md                                   # This file
```

## Detailed Directory Explanations


### 📊 `data/`
Organized storage for project data files:
- `Test/`: the test data, image only
- `Train/`: the training data
- `Meta.csv/`: different attributes of images, including class ID (label), color ID and shape ID.
- `Train.csv/`: attributes of the training images
- `Test.csv/`: attributes of the testing images, including the class ID.


### 📓 `results/`
Result of explorations and analysis.
- `A visualization/`: Contains a visualization of the first layer filters
- `impact of data preprocessing and batch normalization`: A table showing the power of data preprocessing and batch normalization on test accuracy
- `overfitting and underfitting/`: Two heatmaps demonstrating overfitting and underfitting


### Example Usage

## Setup Instructions

1. Clone this repository
```bash
git clone https://github.com/blackhole-hope123/BDAA-final-project.git
cd C:\Users\your-username\BDAA-final-projectpath_to_cloned_repo
```

2. Install the packages
```bash
pip3 install -r requirements.txt
```

#### Running Scripts
for example, to run the main program:
```bash
python main.py
```

## Acknowledgement 
The dataset is from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

"# traffic-sign-recognition-project" 
