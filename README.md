# ComLMEss
An Deep Learning Model For Predicting Essential Protein Task

This repository contains the PyTorch implementation of ComLMEss.

# Requirements
    python==3.9.19
    torch==1.12.1
    numpy==1.26.4
    scikit-learn==1.4.2
    tqdm==4.66.4


# Usage

## Data

The train and test data, which contain different protein information extracted by OntoProtein, ProtTrans, and ESMFold, can be downloaded from ..path..

## Simple usage
After making sure that the features extracted by OntoProtein, ProtTrans, ESMFfold are in **/data** folder, you can train the model in a very simple way by the command blow:

``python train.py ``

## How to  use your own data
First, you need to store your extracted feature files in the **/data** folder.

Second, you need to modify the function **load_feature** and **load_all_features** in the **utils.py** file.

## How to train your own model
With the data ready, you can train your own model by modifying the **class DefaultConfig** in the **utils.py** file.

>In the **class DefaultConfig**, the meaning of the variables is explained as follows:
>>***seed*** is The random seed used to ensure reproducibility of the results. 
>>***kfold*** is the number of folds in cross-validation.  
>>***patience*** is the number of epochs with no improvement after which training will be stopped.  
>>***lr*** is the learning rate, which controls the step size during gradient descent.  
>>***batch_size*** is the number of samples processed before the model is updated.  
>>***dropout*** is the dropout rate, which prevents overfitting by randomly setting a fraction of input units to zero.  
>>***filter*** is the number of filters in the convolutional layers.
>>***activation*** is the activation function used in the model, here 'relu'.  
>>***optimizer*** is the optimization algorithm used for training, here 'Adam'.
>>***kernel_size_onto*** is the The size of the convolutional kernel used for OntoProtein features.
>>***kernel_size_prot*** is the size of the convolutional kernel used for ProtTrans features.
>>***kernel_size_esm*** is the size of the convolutional kernel used for ESMFold features.  
>>***T*** is the cycle length for cosine annealing in the learning rate schedule.

Then, you can train the model by executing the following command:
``python train.py ``



## Other details
The other details can be seen in the paper and the codes.

# Citation

# License
