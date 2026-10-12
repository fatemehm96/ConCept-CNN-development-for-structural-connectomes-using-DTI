# ConCept-CNN-development-for-structural-connectomes-using-DTI
ConCeptCNN: a CNN for classifying frontotemporal dementia (FTD), Alzheimer's disease (AD), and cognitively normal (CN) subjects from DTI-based structural connectomes, with a leakage-free cross-validation pipeline.

ConCeptCNN: Structural Connectome Classification of FTD, AD, and CN

This repository contains the code for classifying frontotemporal dementia (FTD), Alzheimer's disease (AD), and cognitively normal (CN) individuals from diffusion tensor imaging (DTI)-derived structural connectomes using ConCeptCNN, a convolutional neural network designed for connectivity matrices.

Highlights
Task: three-class classification (FTD vs. AD vs. CN) from structural brain networks
Model: ConCeptCNN, a CNN architecture for connectome matrices
Leakage-free evaluation: the train/test split is performed before any class-imbalance augmentation (SMOTE), so synthetic samples never contaminate the test data
Robust validation: 5-fold stratified cross-validation
Proper preprocessing: MinMaxScaler is fit on training folds only and then applied to the held-out fold
Scripts included for training, evaluation, and visualization
Environment
Python 3.9
TensorFlow 2.10.0



