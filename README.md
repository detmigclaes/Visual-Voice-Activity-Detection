# Visual-Voice-Activity-Detection
### Project for Computer Vision F23 - Group 12

This README will cover code used to train and evaluate the model discussed in the report. 

## Dataset

The dataset used to train the model can be downloaded from the following link:
https://www.dropbox.com/s/70jyy3d3hdumbpx/End-to-End-VAD_data.rar?dl=0&fbclid=IwAR2BAZZoxfW1bcEFVpwV4uoVYfeJVrP81R1CX-rHNsrdQbQgU66xT1rrKpA         
The data is comprised of 11 different speakers that can be split into training, validation, test sets. 

## Included Python files

#### train.py
This is the file used for the training loop. This files can be run with different arguments to modify how the model will be trained and which layers and branches will be activated. 

#### eval.py
This file is used to evaluate a models performance on a test dataset. The pretrained model is given to eval.py
We added several metrics that will be exported in a csv file and an svg image with a name that is given as --name=''. 

#### utils
This folder contains three files used for logging the performance and time metrics for the models and the script used to inject the noise samples onto the speech samples. 

#### params 
This folder contains the parameters fror the model that makes it compatible with the dataset that we use.

#### networks
This folder contains the different parts of the model and the different branches. 

## Running the model
Many of the functions used for training and the model depend on older, now depricated, features of the different libraries used. 
The versions listed by the original authors were not compatible with each other, but the following combination worked.

Python==3.9.12

sk_video==1.1.10    
torch==1.7.1+cu110      
numpy==1.23.5       
librosa==0.10.0     
matplotlib==2.2.5       
torchvision==0.8.2+cu110      
scipy==1.10.1         
scikit_learn==1.2.2       
tensorboardX==1.6       
pillow==6.1.0       

The dataset is automaticly generated by the included files. 
