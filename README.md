### Authors
Robin Byl,
Florence Franchomme,
Nédid Ismaili,
Adeline Wantiez

# Image Inpainting
This project aims at inpainting missing region(s) of an image based on the remaining pixels using deep learning-based methods. 
For this, a Generative Adversarial Network (GAN) has been implemented. Its purpose is to generate visually plausible image to fill the missing content. 
The considered dataset is the Animal Faces-HQ (AFHQ).  

To run the code go to Usage section.

## Description
![alt-text](Results/General_architecture.png "Architecture")  
## Results
Results when feeding the whole image to the discriminator:  
![alt-text](Results/Results_All.png "Results")  
Results when feeding the patch to the discriminator:    
![alt-text](Results/Results_patch.png "Results")  
## Files
    -dataset
        -afhq
            -cat
            -dog
            -wild
    -Model
        Generator.py
        Discriminator.py
    -Results
        General_architecture.png
        ResultsExample.png
    Final_figures.py
    Preprocessing.py
    read_data.py
    README.md
    Training.py
    TrainigFunctions.py
    useTrainedModels.py

## Clone Project

git clone https://github.com/FFrancho99/Project_ML.git

## Packages

    numpy >= 1.23.5
    matplotlib >= 3.7.1
    torch >= 2.0.0
    torchmetrics >= 0.11.4
    torchvision >= 0.15.1
    uuid >= 1.30

## Usage
Cuda is required to run the code but if you don't have it, change line 49 of Training.py 'cuda:0' to 'cpu'.  
Run Training.py file to create, train, validate and test the model. If you already have a saved model then run useTrainedModels.py.



## Sources
Dataset: Animal Faces-HQ (AFHQ) from https://www.kaggle.com/datasets/andrewmvd/animal-faces
    
