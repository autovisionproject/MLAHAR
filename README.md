# Enhancing Cross-User Generalization in Human Activity Recognition with Adaptive Noise Scheduling and Linear Attention

## Usage:
This repository contains the implementation of the MLA based HAR mechanism using PyTorch. The code is organized as follows:
- `models/MLAHAR.py`: Implementation of the MLAHAR module.
- `models/train.py`: Script for training and evaluating the FFA-based human activity recognition model.
- `utills/prepareDataset.py`: Script for data preprocessing.
- `utills/graph.py`: Script for generating performance matrices.
- `utills/logger.py`: Script to save model and performance matrices.
- `main.py`:  Final script to run overall model.
## Training 

```
-------------------------------------------------------------
##Update config.yaml file as per target dataset 
    Example :
    skoda:
        log_path: '/home/block/HAR/log/pamap2/'  ----> Update path
        path: '/home/block/HAR/database/' ---> Update dataset path.

##Input format:
--------------------------------------------------------
(myenv) block@l40s:~/HAR/Linear$ python main.py --h
usage: main.py [-h] [--dsname DSNAME] [--epoch EPOCH] [--seed SEED]
Process a dataset based on the dataset name.
options:
  -h, --help       show this help message and exit
  --dsname DSNAME  Dataset name (choose from 'opportunity', 'realdisp', 'pamap2', 'mhealth')
  --epoch EPOCH    Select epoch range (choose from 0-N)
  --seed SEED      Set Seed value [42,24, etc]
(myenv) block@l40s:~/HAR/Linear$

##Run code:
----------------------------------------------------
 #python main.py --dsname <datasetname>   --epoch <epoch> --seed <seed-value>
```
### Requirements
- `pandas version`: 2.0.3
- `torch version`: 2.0.1+cu117
- `numpy version`: 1.24.3
- `scipy version`: 1.10.1
- `matplotlib version`: 3.7.3
- `seaborn version`: 0.12.2
- `Python version`: 3.8.10 
- `fvcore.nn version`: 0.1.5.post20221221
- `sklearn version`: 1.3.1
- `thop`:0.1.1-2209072238

## Confusion Matrix 

### MHEALTH Dataset
| Class-Wise Accuracy | Class-Wise Prediction |
|:--------------:|:----------------:|
| <img src="images/mhealth_accuracy_matrix.png" width="400"> | <img src="images/mhealth_Confusion_matrix.png" width="400"> |
### PAMAP2 Dataset
| Class-Wise Accuracy | Class-Wise Prediction |
|:--------------:|:----------------:|
| <img src="images/pamap2_accuracy_matrix.png" width="400"> | <img src="images/pamap2_Confusion_matrix.png" width="400"> |
### OPPORTUNITY Dataset
| Class-Wise Accuracy | Class-Wise Prediction |
|:--------------:|:----------------:|
| <img src="images/opportunity_accuracy_matrix.png" width="400"> | <img src="images/opportunity_Confusion_matrix.png" width="400"> |
### REALDISP Dataset
| Class-Wise Accuracy | Class-Wise Prediction |
|:--------------:|:----------------:|
| <img src="images/realdisp_accuracy_matrix.png" width="400"> | <img src="images/realdisp_Confusion_matrix.png" width="400"> |

## t-SNE Representation of Learned Features
| MHEALTH Dataset | OPPORTUNITY Dataset | PAMAP2 Dataset | REALDISP Dataset |
|:--------------:|:----------------:|:--------------:|:--------------:|
| <img src="images/mhealth_tnse_pre.png" width="200"> | <img src="images/opp_tnse_pre.png" width="200"> | <img src="images/pamap2_tnse_pre.png" width="200"> | <img src="images/realdisp_tnse_pre.png" width="200"> |

## Model Generalization Analysis Across Test Subsets
| Accuracy (%) | F1-weighted (%) | F1-macro (%) |
|:--------:|:----------:|:--------:|
| <img src="images/Accuracy.png" width="220"> | <img src="images/F1w.png" width="220"> | <img src="images/F1m.png" width="220"> |


